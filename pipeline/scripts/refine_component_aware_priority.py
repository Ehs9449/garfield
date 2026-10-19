#!/usr/bin/env python3
"""Generalized component-aware semantic refinement.

Purpose
-------
Prevent large/background semantic classes (especially wall/roof) from suppressing
smaller building components that SAM3 detects correctly.

This preserves the successful viewer-only strict behavior:
- reuse existing viewer-path RGBs and existing multiclass SAM3 masks
- evaluate a component only in views where SAM3 detected that component
- erode masks to reduce surrounding-wall contamination
- z-buffer projected 3D points
- require multi-view support
- refine semantics pointwise only; NEVER promote a whole K400 cluster

Generalization
--------------
Localized components are evaluated independently:
    opening, beam, column, door, window

Each component gets normalized evidence:
    support_ratio = hits / opportunities
    mean_confidence = accumulated SAM3 confidence / hits
    evidence_score = support_ratio * mean_confidence

A component may override the broad Option-B label only when:
1) its multi-view support thresholds are satisfied,
2) its current/base semantic class is eligible for refinement, and
3) it wins component-vs-component competition at that point.

Large broad classes therefore do not compete by raw accumulated view count.
Window has highest priority only as a tie-breaker, not as an unconditional override.

No SAM3 inference, GARField training, rendering, or clustering is rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

COLORS = {
    'window': (0,205,255), 'wall': (179,179,153), 'roof': (204,51,51),
    'door': (204,102,0), 'opening': (255,153,0), 'column': (230,230,230),
    'beam': (153,76,0), 'ceiling': (204,204,230), 'floor': (128,128,102),
    'curtain wall': (77,153,204), 'ground': (102,102,102),
    'vegetation': (0,153,0), 'sky': (128,179,255),
    'staircase': (255,255,0), 'railing': (179,76,179), 'unknown': (75,75,75),
}

# Low -> high tie-break priority. Evidence score decides first.
# Window is intentionally last/highest as requested.
COMPONENTS = ('opening', 'beam', 'column', 'door', 'window')

# Components may refine broad facade labels and closely related component labels.
# Ground / vegetation / sky / floor / ceiling / staircase / railing are protected.
ALLOWED = {
    'window': {
        'wall', 'roof', 'curtain wall', 'opening', 'door', 'window', 'unknown'
    },
    'door': {
        'wall', 'curtain wall', 'opening', 'door', 'window', 'unknown'
    },
    'opening': {
        'wall', 'curtain wall', 'opening', 'door', 'window', 'unknown'
    },
    'column': {
        'wall', 'roof', 'curtain wall', 'column', 'beam', 'opening', 'unknown'
    },
    'beam': {
        'wall', 'roof', 'curtain wall', 'beam', 'column', 'opening', 'unknown'
    },
}

# Thin structural components can disappear under a 2-pixel erosion, so column/beam
# default to one pixel. Window/door/opening preserve the successful strict 2-pixel erosion.
DEFAULT_EROSION = {
    'opening': 2,
    'beam': 1,
    'column': 1,
    'door': 2,
    'window': 2,
}


def project_points(xyz, rec):
    transform = np.asarray(rec['c2w'], np.float64)
    if transform.shape != (3, 4):
        raise ValueError(f"Invalid pose for {rec['view_name']}: {transform.shape}")
    T = np.eye(4)
    T[:3] = transform
    C = np.linalg.inv(T)
    cam = xyz @ C[:3, :3].T + C[:3, 3]
    depth = -cam[:, 2]
    ids = np.flatnonzero(depth > 0.01)
    fx, fy, cx, cy = (float(rec[k]) for k in ('fx','fy','cx','cy'))
    u = np.rint(fx * cam[ids,0] / depth[ids] + cx).astype(np.int32)
    v = np.rint(-fy * cam[ids,1] / depth[ids] + cy).astype(np.int32)
    W, H = int(rec['img_w']), int(rec['img_h'])
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return ids[ok], u[ok], v[ok], depth[ids[ok]]


def zbuffer(ids, u, v, depth, W, H, tile=4, abs_tol=.02, rel_tol=.025):
    if len(ids) == 0:
        return ids, u, v
    nx = (W + tile - 1) // tile
    bin_id = (v // tile) * nx + (u // tile)
    closest = np.full(nx * ((H + tile - 1)//tile), np.inf, np.float32)
    np.minimum.at(closest, bin_id, depth.astype(np.float32))
    near = closest[bin_id]
    ok = depth <= near + np.maximum(abs_tol, rel_tol * near)
    return ids[ok], u[ok], v[ok]


def write_ply(path, xyz, semantic):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.empty((len(semantic), 3), np.uint8)
    rgb[:] = COLORS['unknown']
    for cls in np.unique(semantic):
        rgb[semantic == cls] = COLORS.get(str(cls), COLORS['unknown'])

    arr = np.empty(len(xyz), dtype=[
        ('x','<f4'),('y','<f4'),('z','<f4'),
        ('red','u1'),('green','u1'),('blue','u1')
    ])
    for i,k in enumerate('xyz'):
        arr[k] = xyz[:, i]
    for i,k in enumerate(('red','green','blue')):
        arr[k] = rgb[:, i]

    head = (
        'ply\nformat binary_little_endian 1.0\n'
        + f'element vertex {len(xyz)}\n'
        + 'property float x\nproperty float y\nproperty float z\n'
        + 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        + 'end_header\n'
    )
    with path.open('wb') as f:
        f.write(head.encode())
        arr.tofile(f)


def parse_overrides(text, cast, valid_components):
    """Parse strings like window=3,door=2,beam=2."""
    result = {}
    if not text:
        return result
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f'Invalid override "{item}" (expected class=value)')
        name, value = item.split('=', 1)
        name = name.strip()
        if name not in valid_components:
            raise ValueError(f'Unknown component in override: {name}')
        result[name] = cast(value)
    return result


def support(hits, opportunities, min_views, min_ratio):
    ratio = hits.astype(np.float32) / np.maximum(opportunities, 1)
    return (hits >= min_views) & (ratio >= min_ratio), ratio


def run(a):
    required = [a.points, a.viewer_labels, a.views/'view_params.json']
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    if not a.masks.is_dir():
        raise FileNotFoundError(a.masks)

    xyz = np.load(a.points).astype(np.float32)
    base = np.load(a.viewer_labels, allow_pickle=False).astype('<U32')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or base.shape != (len(xyz),):
        raise ValueError('points.npy and semantic label array must have identical point order/length')

    records = json.loads((a.views/'view_params.json').read_text())
    if not records:
        raise ValueError('No camera records in view_params.json')

    min_views = {c: a.min_views for c in COMPONENTS}
    min_ratio = {c: a.min_ratio for c in COMPONENTS}
    erosion = dict(DEFAULT_EROSION)

    min_views.update(parse_overrides(a.min_views_overrides, int, COMPONENTS))
    min_ratio.update(parse_overrides(a.min_ratio_overrides, float, COMPONENTS))
    erosion.update(parse_overrides(a.erosion_overrides, int, COMPONENTS))

    for c in COMPONENTS:
        if min_views[c] < 1:
            raise ValueError(f'{c}: min_views must be >= 1')
        if not 0 <= min_ratio[c] <= 1:
            raise ValueError(f'{c}: min_ratio must be in [0,1]')
        if erosion[c] < 0:
            raise ValueError(f'{c}: erosion must be >= 0')

    n = len(xyz)
    hits = {c: np.zeros(n, np.uint16) for c in COMPONENTS}
    opp = {c: np.zeros(n, np.uint16) for c in COMPONENTS}
    confidence_sum = {c: np.zeros(n, np.float32) for c in COMPONENTS}

    views_with_masks = Counter()
    view_rows = []

    for i, rec in enumerate(records, 1):
        name = rec['view_name']
        mask_file = a.masks/(name + '_masks.npz')
        rgb_file = a.views/rec.get('image_file', name + '.jpg')
        if not mask_file.is_file() or not rgb_file.is_file():
            raise FileNotFoundError(f'Missing image or mask file for {name}')

        W, H = int(rec['img_w']), int(rec['img_h'])
        score_maps = {}
        kept_counts = {}

        with np.load(mask_file, allow_pickle=False) as data:
            if '_source_sha256' in data:
                got = str(data['_source_sha256'].item())
                want = hashlib.sha256(rgb_file.read_bytes()).hexdigest()
                if got != want:
                    raise ValueError(f'{name}: SAM3 mask file is stale relative to JPEG')

            for c in COMPONENTS:
                key = c.replace(' ', '_')
                mk, sk = key+'_masks', key+'_scores'
                if mk not in data or sk not in data:
                    raise ValueError(
                        f'{mask_file} has no {c} masks. Existing multiclass SAM3 '
                        f'inference must include prompt "{c}".'
                    )

                masks, scores = data[mk], data[sk]
                if masks.ndim != 3 or masks.shape[1:] != (H,W) or len(masks) != len(scores):
                    raise ValueError(f'{name}: invalid {c} mask shape')

                score_map = np.zeros((H,W), np.float32)
                kept = 0
                for m,s in zip(masks, scores):
                    s = float(s)
                    if s < a.min_sam3_score or not m.any():
                        continue

                    eroded = m.astype(bool)
                    if erosion[c] > 0:
                        eroded = binary_erosion(eroded, iterations=erosion[c])
                    if not eroded.any():
                        continue

                    kept += 1
                    # Max, not sum: overlapping masks in one view cannot artificially
                    # multiply evidence for that class.
                    score_map[eroded] = np.maximum(score_map[eroded], s)

                score_maps[c] = score_map
                kept_counts[c] = kept

        active = [c for c in COMPONENTS if kept_counts[c] > 0]
        visible = 0
        class_hits = {c:0 for c in COMPONENTS}

        if active:
            idx,u,v,depth = project_points(xyz, rec)
            idx,u,v = zbuffer(idx,u,v,depth,W,H,a.tile,a.abs_tol,a.rel_tol)
            visible = len(idx)

            for c in active:
                views_with_masks[c] += 1

                # Critical normalization:
                # opportunity exists only in a view where this component was actually
                # detected by SAM3. Broad wall/roof detections therefore cannot gain
                # extra votes simply by appearing in more views.
                opp[c][idx] += 1

                projected_scores = score_maps[c][v,u]
                matched = projected_scores > 0
                h = idx[matched]

                hits[c][h] += 1
                confidence_sum[c][h] += projected_scores[matched]
                class_hits[c] = len(h)

        row = {'view':name, 'zbuffer_visible':visible}
        for c in COMPONENTS:
            row[f'{c}_masks'] = kept_counts[c]
            row[f'{c}_point_hits'] = class_hits[c]
        view_rows.append(row)

        print(
            f'[{i}/{len(records)}] {name}: visible={visible:,} '
            + ' '.join(
                f'{c}[m={kept_counts[c]},hit={class_hits[c]:,}]'
                for c in COMPONENTS
            ),
            flush=True
        )

    supported = {}
    eligible = {}
    ratios = {}
    mean_conf = {}
    evidence = {}

    for c in COMPONENTS:
        supported[c], ratios[c] = support(
            hits[c], opp[c], min_views[c], min_ratio[c]
        )
        eligible[c] = np.isin(base, list(ALLOWED[c]))
        supported[c] &= eligible[c]

        mean_conf[c] = np.divide(
            confidence_sum[c],
            np.maximum(hits[c], 1),
            dtype=np.float32
        )

        # Primary competition is normalized support, modulated by mask confidence.
        # No raw wall/roof count enters this score.
        evidence[c] = ratios[c] * mean_conf[c]
        evidence[c][~supported[c]] = -1.0

    # Component-vs-component competition.
    # Highest evidence wins. A very small epsilon establishes deterministic tie
    # priority: opening < beam < column < door < window.
    score_stack = []
    for rank, c in enumerate(COMPONENTS):
        score_stack.append(evidence[c] + rank * 1e-7)
    score_stack = np.stack(score_stack, axis=0)

    best_component_idx = np.argmax(score_stack, axis=0)
    best_score = np.max(score_stack, axis=0)
    any_component = best_score >= 0

    refined = base.copy()
    winner_counts = Counter()
    for idx_c, c in enumerate(COMPONENTS):
        sel = any_component & (best_component_idx == idx_c)
        refined[sel] = c
        winner_counts[c] = int(sel.sum())

    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    evidence_dir = out/'evidence'
    evidence_dir.mkdir(exist_ok=True)

    for c in COMPONENTS:
        np.save(evidence_dir/f'{c}_hits.npy', hits[c])
        np.save(evidence_dir/f'{c}_opportunities.npy', opp[c])
        np.save(evidence_dir/f'{c}_support_ratio.npy', ratios[c])
        np.save(evidence_dir/f'{c}_mean_confidence.npy', mean_conf[c])
        np.save(evidence_dir/f'{c}_evidence_score.npy', evidence[c])
        np.save(evidence_dir/f'{c}_strict_supported.npy', supported[c])

    np.save(out/'semantic_labels.npy', refined)
    np.save(out/'winning_component_index.npy', best_component_idx.astype(np.int8))
    np.save(out/'winning_component_score.npy', best_score.astype(np.float32))
    write_ply(out/'semantic_pointcloud.ply', xyz, refined)

    per_class = out/'per_class'
    per_class.mkdir(exist_ok=True)
    class_counts = Counter(refined.tolist())

    for c,count in sorted(class_counts.items()):
        sel = refined == c
        write_ply(per_class/f'{c.replace(" ", "_")}.ply', xyz[sel], refined[sel])

    for c in COMPONENTS:
        sel = refined == c
        if sel.any():
            write_ply(out/f'{c}_only.ply', xyz[sel], refined[sel])

    with (out/'per_view_component_evidence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
        writer.writeheader()
        writer.writerows(view_rows)

    with (out/'class_counts.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['semantic_class','points'])
        for c,count in sorted(class_counts.items()):
            w.writerow([c,count])

    with (out/'component_summary.csv').open('w', newline='') as f:
        fields = [
            'component','views_with_masks','supported_before_competition',
            'final_winner_points','min_views','min_ratio','erosion',
            'max_hits','max_opportunities','mean_supported_ratio',
            'mean_supported_confidence'
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in COMPONENTS:
            sup = supported[c]
            w.writerow({
                'component': c,
                'views_with_masks': int(views_with_masks[c]),
                'supported_before_competition': int(sup.sum()),
                'final_winner_points': int(winner_counts[c]),
                'min_views': int(min_views[c]),
                'min_ratio': float(min_ratio[c]),
                'erosion': int(erosion[c]),
                'max_hits': int(hits[c].max()) if len(hits[c]) else 0,
                'max_opportunities': int(opp[c].max()) if len(opp[c]) else 0,
                'mean_supported_ratio': float(ratios[c][sup].mean()) if sup.any() else 0.0,
                'mean_supported_confidence': float(mean_conf[c][sup].mean()) if sup.any() else 0.0,
            })

    summary = {
        'points': n,
        'camera_views': len(records),
        'base_optionB_class_counts': dict(Counter(base.tolist())),
        'final_class_counts': dict(class_counts),
        'config': {
            'components_low_to_high_tie_priority': list(COMPONENTS),
            'highest_tie_priority': COMPONENTS[-1],
            'min_sam3_score': a.min_sam3_score,
            'min_views': min_views,
            'min_ratio': min_ratio,
            'erosion_pixels': erosion,
            'evidence_formula': 'support_ratio * mean_sam3_confidence',
            'opportunities_only_in_views_where_component_has_masks': True,
            'point_level_only_no_cluster_promotion': True,
            'broad_classes_do_not_compete_by_raw_view_count': True,
        },
        'component_stats': {}
    }

    for c in COMPONENTS:
        sup = supported[c]
        summary['component_stats'][c] = {
            'views_with_masks': int(views_with_masks[c]),
            'supported_points_before_component_competition': int(sup.sum()),
            'final_winner_points': int(winner_counts[c]),
            'max_hits': int(hits[c].max()) if len(hits[c]) else 0,
            'max_opportunities': int(opp[c].max()) if len(opp[c]) else 0,
            'mean_supported_ratio': float(ratios[c][sup].mean()) if sup.any() else 0.0,
            'mean_supported_confidence': float(mean_conf[c][sup].mean()) if sup.any() else 0.0,
        }

    (out/'summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== DONE: generalized component-aware strict matching ===')
    print('Base Option-B:', dict(Counter(base.tolist())))
    for c in COMPONENTS:
        print(
            f'{c:8s}: views_with_masks={views_with_masks[c]:3d}, '
            f'supported={int(supported[c].sum()):,}, '
            f'final={winner_counts[c]:,}'
        )
    print('Final:', dict(class_counts))
    print('Semantic PLY:', out/'semantic_pointcloud.ply')
    print('Per-class PLYs:', per_class)
    print('Component summary:', out/'component_summary.csv')
    print('Summary:', out/'summary.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path('outputs/towerlsu')
    exp = root/'feature_clustering_experiments'

    p.add_argument(
        '--points', type=Path,
        default=root/'ortho_projection_guided_plus_top/points.npy'
    )
    p.add_argument(
        '--viewer-labels', type=Path,
        default=exp/'22_viewer_path_full_multiclass/semantic/semantic_labels_pointwise.npy'
    )
    p.add_argument(
        '--views', type=Path,
        default=root/'viewer_path_experiment/full/views'
    )
    p.add_argument(
        '--masks', type=Path,
        default=root/'viewer_path_experiment/multiclass/masks'
    )
    p.add_argument(
        '--output', type=Path,
        default=exp/'25_viewer_component_aware/viewer_only_strict'
    )

    p.add_argument('--min-sam3-score', type=float, default=.30)
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--min-ratio', type=float, default=.25)

    p.add_argument(
        '--min-views-overrides', default='',
        help='Optional class overrides, e.g. "door=2,column=2,beam=2"'
    )
    p.add_argument(
        '--min-ratio-overrides', default='',
        help='Optional class overrides, e.g. "door=.20,column=.20,beam=.20"'
    )
    p.add_argument(
        '--erosion-overrides', default='',
        help='Optional class overrides, e.g. "window=2,door=2,column=1,beam=1"'
    )

    p.add_argument('--tile', type=int, default=4)
    p.add_argument('--abs-tol', type=float, default=.02)
    p.add_argument('--rel-tol', type=float, default=.025)

    a = p.parse_args()
    if a.min_views < 1 or not (0 <= a.min_ratio <= 1):
        p.error('Invalid threshold')
    run(a)


if __name__ == '__main__':
    main()
