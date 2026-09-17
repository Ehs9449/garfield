#!/usr/bin/env python3
"""
Data access layer for the pipeline control panel.

Everything that touches the filesystem lives here, so app.py stays about
layout.  Nothing in this module imports streamlit, so it can be tested on
its own:

    python pipeline/ui/pipeline_io.py         # prints a status report
"""

from __future__ import annotations

import colorsys
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# The repository root: this file is <root>/pipeline/ui/pipeline_io.py
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = REPO_ROOT / "pipeline" / "config.yaml"
SNAKEFILE = REPO_ROOT / "pipeline" / "Snakefile"

UI_DIR = REPO_ROOT / "outputs" / "ui"
LOG_DIR = UI_DIR / "logs"
HISTORY_PATH = UI_DIR / "history.jsonl"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path: Path = CONFIG_PATH) -> Path:
    """Write the config back, keeping a timestamped backup.

    Uses ruamel.yaml when it is installed, because that keeps your comments.
    Falls back to PyYAML, which does not.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(f".yaml.bak_{stamp}")
    if path.exists():
        shutil.copy2(path, backup)

    try:
        from ruamel.yaml import YAML

        ruamel = YAML()
        ruamel.preserve_quotes = True
        ruamel.width = 4096
        with open(path) as f:
            doc = ruamel.load(f)
        _deep_update(doc, cfg)
        with open(path, "w") as f:
            ruamel.dump(doc, f)
    except ImportError:
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    return backup


def comments_preserved() -> bool:
    try:
        import ruamel.yaml  # noqa: F401

        return True
    except ImportError:
        return False


def _deep_update(target, source):
    """Copy values from source into target without dropping unknown keys."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

def stage_paths(cfg: dict) -> dict:
    """Rebuild the same paths the Snakefile builds, from the same config."""
    name = cfg["dataset"]["name"]
    base = REPO_ROOT / "outputs" / name

    def under(section, key, default):
        raw = cfg.get(section, {}).get(key, default)
        return base / Path(str(raw)).name

    return {
        "dataset_name": name,
        "dataset_path": Path(cfg["dataset"]["path"]),
        "base": base,
        "pointcloud": base / "exports"
        / Path(str(cfg["projection"]["pointcloud"])).name,
        "proj": under("projection", "output_dir", "ortho_projection"),
        "cluster": under("clustering", "output_dir", "clustering"),
        "merge": under("merging", "output_dir", "merged"),
        "views": under("labeling_views", "output_dir", "labeling_views"),
        "masks": base / "labeling_masks_finetuned",
        "semantic": under("matching", "output_dir", "semantic_labels"),
    }


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

STAGES = [
    {
        "key": "ortho_projection",
        "label": "Ortho projection",
        "note": "Renders the side and roof views and gives every point a GARField feature.",
        "target": lambda p: p["proj"] / "avg_features.npy",
        "gpu": True,
    },
    {
        "key": "clustering",
        "label": "HDBSCAN clustering",
        "note": "Groups the points by their features. Optuna searches the parameters.",
        "target": lambda p: p["cluster"] / "cluster_labels.npy",
        "gpu": True,
    },
    {
        "key": "cluster_pair_evidence",
        "label": "SAM2 pair evidence",
        "note": "Finds cluster pairs that share a SAM2 mask in the facade-facing cameras.",
        "target": lambda p: p["cluster"] / "sam2_cluster_pair_evidence.csv",
        "gpu": False,
    },
    {
        "key": "merge_candidates",
        "label": "Merge candidates",
        "note": "Scores those pairs with GARField feature similarity.",
        "target": lambda p: p["cluster"] / "merge_candidates.csv",
        "gpu": False,
    },
    {
        "key": "merge_clusters",
        "label": "Merge clusters",
        "note": "Joins the pairs that pass both the support and the cosine test.",
        "target": lambda p: p["merge"] / "merged_labels.npy",
        "gpu": False,
    },
    {
        "key": "render_views",
        "label": "Labeling views",
        "note": "Renders the camera rings around the building for SAM3.",
        "target": lambda p: p["views"] / "view_params.json",
        "gpu": True,
    },
    {
        "key": "sam3_inference",
        "label": "SAM3 inference (HPC)",
        "note": "Uploads the views to the HPC and runs the fine-tuned SAM3. Needs SSH.",
        "target": lambda p: p["base"] / ".sam3_inference_done",
        "gpu": False,
    },
    {
        "key": "semantic_labeling",
        "label": "Semantic labeling",
        "note": "Matches merged clusters to SAM3 masks and writes the final point cloud.",
        "target": lambda p: p["semantic"] / "semantic_pointcloud.ply",
        "gpu": False,
    },
]


def stage_status(cfg: dict) -> list:
    paths = stage_paths(cfg)
    rows = []
    for stage in STAGES:
        target = stage["target"](paths)
        exists = target.exists()
        rows.append(
            {
                "key": stage["key"],
                "label": stage["label"],
                "note": stage["note"],
                "target": target,
                "done": exists,
                "mtime": (
                    datetime.fromtimestamp(target.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    if exists
                    else ""
                ),
                "size_mb": round(target.stat().st_size / 1e6, 2) if exists else 0.0,
            }
        )
    return rows


def snakemake_command(rule: str, target: Path, dry: bool = False,
                      force: bool = True, prefix: str = "") -> list:
    cmd = [
        "snakemake",
        "-s",
        str(SNAKEFILE.relative_to(REPO_ROOT)),
        "--cores",
        "1",
        "--rerun-triggers",
        "mtime",
    ]
    if rule:
        cmd += ["--allowed-rules", rule]
    if force and not dry:
        cmd += ["--force"]
    if dry:
        cmd += ["-n"]
    cmd += [str(Path(target).relative_to(REPO_ROOT))]

    if prefix.strip():
        return prefix.split() + cmd
    return cmd


def new_log_path(rule: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{stamp}_{rule}.log"


# ----------------------------------------------------------------------
# Reading results
# ----------------------------------------------------------------------

def read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_view_stats(proj_dir: Path):
    """Per-view statistics written by the projection script.

    Falls back to reading the newest snakemake log, so runs made before the
    script started writing view_stats.json still show something.
    """
    stats = read_json(proj_dir / "view_stats.json")
    if stats:
        return stats, "view_stats.json"

    parsed = _parse_views_from_logs()
    if parsed:
        return parsed, "snakemake log"
    return None, None


_VIEW_RE = re.compile(r"--- View \d+/\d+: (\S+) ---")
_VIS_RE = re.compile(r"Visible points: (\d+) \(([\d.]+)%\)")
_SIZE_RE = re.compile(r"Rendered: (\d+)x(\d+)x\d+ features")


def _parse_views_from_logs(limit: int = 6):
    logs = sorted(
        list((REPO_ROOT / ".snakemake" / "log").glob("*.log"))
        + list(LOG_DIR.glob("*.log")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    for log in logs:
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue

        views, name, size = [], None, None
        for line in text.splitlines():
            m = _VIEW_RE.search(line)
            if m:
                name = m.group(1)
                size = None
                continue
            m = _SIZE_RE.search(line)
            if m and name:
                size = [int(m.group(2)), int(m.group(1))]  # width, height
                continue
            m = _VIS_RE.search(line)
            if m and name:
                views.append(
                    {
                        "name": name,
                        "visible_points": int(m.group(1)),
                        "visible_pct": float(m.group(2)),
                        "img_width": size[0] if size else None,
                        "img_height": size[1] if size else None,
                    }
                )
                name = None
        if views:
            return {"views": views, "source_log": log.name}
    return None


def view_balance(stats: dict) -> dict:
    """How much of the averaged feature comes from side views vs top views."""
    views = stats.get("views", [])
    side = sum(v["visible_points"] for v in views if v["name"].startswith("side"))
    top = sum(v["visible_points"] for v in views if not v["name"].startswith("side"))
    total = side + top
    return {
        "n_views": len(views),
        "side_samples": side,
        "top_samples": top,
        "side_share": 100.0 * side / total if total else 0.0,
        "top_share": 100.0 * top / total if total else 0.0,
    }


def read_points(paths: dict):
    pts = paths["proj"] / "points.npy"
    return np.load(pts) if pts.exists() else None


def read_labels(path: Path):
    return np.load(path) if path.exists() else None


def coverage(paths: dict):
    """Share of points that were seen by at least one view."""
    f = paths["proj"] / "feature_count.npy"
    if not f.exists():
        return None
    counts = np.load(f)
    seen = int((counts > 0).sum())
    return {
        "n_points": int(len(counts)),
        "with_features": seen,
        "pct": 100.0 * seen / len(counts) if len(counts) else 0.0,
        "mean_views": float(counts[counts > 0].mean()) if seen else 0.0,
    }


def cluster_sizes(labels: np.ndarray):
    valid = labels[labels >= 0]
    ids, counts = np.unique(valid, return_counts=True)
    order = np.argsort(-counts)
    return ids[order], counts[order], float(100.0 * (labels == -1).sum() / len(labels))


def semantic_table(paths: dict):
    data = read_json(paths["semantic"] / "semantic_labels.json")
    if not data:
        return None
    rows = []
    for cid, rec in data.items():
        rows.append(
            {
                "cluster": int(cid),
                "label": rec.get("label", "unknown"),
                "points": rec.get("n_points", 0),
                "confidence": round(float(rec.get("confidence", 0.0)), 3),
                "votes": rec.get("votes", 0),
                "avg_iou": round(float(rec.get("avg_iou", 0.0)), 4),
                "breakdown": ", ".join(
                    f"{k}:{v}" for k, v in (rec.get("vote_breakdown") or {}).items()
                ),
            }
        )
    rows.sort(key=lambda r: -r["points"])
    return rows


def label_totals(rows) -> dict:
    totals = {}
    for r in rows:
        totals[r["label"]] = totals.get(r["label"], 0) + r["points"]
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


# ----------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------

def load_training_poses(paths: dict):
    """The camera-to-world matrices of the COLMAP training cameras."""
    for candidate in (
        paths["base"] / "training_camera_poses.npy",
        REPO_ROOT / "outputs" / "training_camera_poses.npy",
    ):
        if candidate.exists():
            poses = np.load(candidate)
            if poses.ndim == 3 and poses.shape[1] in (3, 4):
                return poses.astype(np.float32), candidate
    return None, None


def camera_angles(poses: np.ndarray, center) -> dict:
    """Viewing angles of each camera, the same way the merge filter sees them."""
    center = np.asarray(center, dtype=np.float32)

    position = poses[:, :3, 3]
    # Nerfstudio/OpenGL cameras look along local -Z.
    look = -poses[:, :3, 2]
    look = look / (np.linalg.norm(look, axis=1, keepdims=True) + 1e-8)

    elevation = np.degrees(np.arcsin(np.clip(-look[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(look[:, 1], look[:, 0])) % 360.0

    to_center = center[None, :] - position
    norm = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-8
    cos_off = np.sum(look * (to_center / norm), axis=1)
    offaxis = np.degrees(np.arccos(np.clip(cos_off, -1.0, 1.0)))

    horiz = look[:, :2]
    hnorm = np.linalg.norm(horiz, axis=1, keepdims=True) + 1e-8

    return {
        "position": position,
        "look": look,
        "elevation": elevation,
        "azimuth": azimuth,
        "offaxis": offaxis,
        "horizontal": horiz / hnorm,
        "distance": norm[:, 0],
    }


def filter_cameras(angles: dict, max_elevation, max_incidence, max_offaxis,
                   facade_azimuths) -> dict:
    """Which cameras the merge stage would keep, and why the others fail."""
    facade_dirs = np.array(
        [[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in facade_azimuths],
        dtype=np.float32,
    )
    cosines = angles["horizontal"] @ facade_dirs.T
    incidence = np.degrees(np.arccos(np.clip(cosines.max(axis=1), -1.0, 1.0)))

    ok_elev = angles["elevation"] <= max_elevation
    ok_inc = incidence <= max_incidence
    ok_off = angles["offaxis"] <= max_offaxis
    keep = ok_elev & ok_inc & ok_off

    reason = np.full(len(keep), "kept", dtype=object)
    reason[~ok_elev] = "too steep"
    reason[ok_elev & ~ok_off] = "not facing the building"
    reason[ok_elev & ok_off & ~ok_inc] = "oblique to the facade"

    return {
        "incidence": incidence,
        "keep": keep,
        "reason": reason,
        "n_kept": int(keep.sum()),
        "n_total": int(len(keep)),
    }


def view_camera_positions(paths: dict):
    """Camera positions of the rendered labeling views."""
    data = read_json(paths["views"] / "view_params.json")
    if not data:
        return None
    rows = []
    for view in data:
        c2w = np.array(view["c2w"], dtype=np.float32)
        rows.append(
            {
                "name": view.get("view_name", ""),
                "azimuth": float(view.get("azimuth", 0.0)),
                "elevation": float(view.get("elevation", 0.0)),
                "position": c2w[:3, 3],
            }
        )
    return rows


# ----------------------------------------------------------------------
# Merge evidence
# ----------------------------------------------------------------------

def load_candidates(paths: dict):
    """The merge candidate pairs with their support and feature similarity."""
    path = paths["cluster"] / "merge_candidates.csv"
    if not path.exists():
        return None
    import csv

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "cluster_a": int(row["cluster_a"]),
                        "cluster_b": int(row["cluster_b"]),
                        "supporting_views": int(row["supporting_views"]),
                        "cosine_similarity": float(row["cosine_similarity"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_evidence(paths: dict):
    """The raw cluster-pair evidence, before feature similarity was added."""
    path = paths["cluster"] / "sam2_cluster_pair_evidence.csv"
    if not path.exists():
        return None
    import csv

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "cluster_a": int(row["cluster_a"]),
                        "cluster_b": int(row["cluster_b"]),
                        "supporting_views": int(row["supporting_views"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def merge_preview(candidates, cluster_ids, min_support, min_cosine) -> dict:
    """Apply the merge rule here, without running the pipeline.

    This is the same union-find the real merging stage uses, so merges chain:
    if 1+2 and 2+3 both pass, the result is one group {1, 2, 3}.

    What it does NOT do - and neither does the pipeline today - is re-measure
    the merged group against a further cluster with fresh evidence. Every
    decision comes from the original pairwise numbers.
    """
    parent = {int(c): int(c) for c in cluster_ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    accepted = []
    for row in candidates or []:
        if (row["supporting_views"] >= min_support
                and row["cosine_similarity"] >= min_cosine):
            a, b = row["cluster_a"], row["cluster_b"]
            if a in parent and b in parent:
                accepted.append(row)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    groups = {}
    for cid in parent:
        groups.setdefault(find(cid), []).append(cid)

    multi = sorted((sorted(g) for g in groups.values() if len(g) > 1),
                   key=len, reverse=True)

    return {
        "accepted_pairs": accepted,
        "n_accepted": len(accepted),
        "groups": multi,
        "n_groups": len(multi),
        "clusters_before": len(parent),
        "clusters_after": len(groups),
        "largest_group": len(multi[0]) if multi else 0,
        "chained": sum(1 for g in multi if len(g) > 2),
    }


# ----------------------------------------------------------------------
# SAM2 cache - the masks the merge evidence is built from
# ----------------------------------------------------------------------

def sam_cache_dir(cfg: dict) -> Path:
    raw = cfg.get("merging", {}).get("sam_cache", "data/sam_cache")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def list_sam_cache(cfg: dict):
    directory = sam_cache_dir(cfg)
    if not directory.exists():
        return []
    return sorted(directory.glob("sam_*.npz"))


def dataset_images(cfg: dict):
    images = Path(cfg["dataset"]["path"]) / "images"
    if not images.exists():
        return []
    return sorted(
        p for p in images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def colorize_groups(keys_2d: np.ndarray) -> np.ndarray:
    """Turn one level of SAM2 pixel_level_keys into a color image."""
    ids = np.unique(keys_2d)
    ids = ids[ids >= 0]
    palette = cluster_palette(max(len(ids), 1))
    lookup = {int(g): palette[i] for i, g in enumerate(ids)}

    out = np.zeros((*keys_2d.shape, 3), dtype=np.uint8)
    for gid, hex_color in lookup.items():
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        out[keys_2d == gid] = rgb
    return out


# ----------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------

# Same class colors the pipeline writes into the PLY files, so the browser
# view and CloudCompare agree.
CLASS_COLORS = {
    "window": "#00ccff",
    "wall": "#b3b399",
    "roof": "#cc3333",
    "door": "#cc6600",
    "opening": "#ff9900",
    "column": "#e6e6e6",
    "beam": "#994d00",
    "ceiling": "#ccccee",
    "floor": "#808066",
    "curtain wall": "#4d99cc",
    "ground": "#666666",
    "vegetation": "#009900",
    "sky": "#80b3ff",
    "staircase": "#ffff00",
    "railing": "#b34db3",
    "unknown": "#4d4d4d",
    "noise": "#333333",
}

# Two-series palette, validated for light and dark surfaces.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
MUTED = "#8a8a84"


def cluster_palette(n: int) -> list:
    """Distinct colors for cluster ids.

    These separate neighbouring clusters visually. They are not an identity
    code - the hover text and the highlight selector carry the cluster id.
    """
    colors = []
    for i in range(max(n, 1)):
        hue = (i * 137.508 % 360.0) / 360.0
        light = 0.58 if i % 2 == 0 else 0.44
        r, g, b = colorsys.hls_to_rgb(hue, light, 0.62)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# ----------------------------------------------------------------------
# Run history
# ----------------------------------------------------------------------

TRACKED = {
    "projection": ["scale", "n_side", "side_elevations", "n_top_ring",
                   "top_elevations", "azimuth_offset", "dist_factor", "resolution"],
    "clustering": ["n_trials", "sample_size", "optimization_min_clusters",
                   "optimization_max_clusters"],
    "merging": ["min_cluster_fraction", "min_support", "min_cosine",
                "max_elevation_deg", "max_incidence_deg", "max_offaxis_deg"],
    "matching": ["min_iou"],
}


def config_snapshot(cfg: dict) -> dict:
    snap = {}
    for section, keys in TRACKED.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            snap[f"{section}.{key}"] = value
    rings = cfg.get("labeling_views", {}).get("rings")
    if rings:
        snap["labeling_views.rings"] = ";".join(
            f"{r['elevation']}@{r['n_azimuth']}" for r in rings
        )
    return snap


def collect_metrics(cfg: dict) -> dict:
    paths = stage_paths(cfg)
    metrics = {}

    cov = coverage(paths)
    if cov:
        metrics["points"] = cov["n_points"]
        metrics["coverage_pct"] = round(cov["pct"], 1)
        metrics["mean_views_per_point"] = round(cov["mean_views"], 2)

    stats, _ = read_view_stats(paths["proj"])
    if stats:
        bal = view_balance(stats)
        metrics["n_views"] = bal["n_views"]
        metrics["side_share_pct"] = round(bal["side_share"], 1)

    opt = read_json(paths["cluster"] / "optimization_results.json")
    if opt:
        final = opt.get("final_metrics", {})
        metrics["n_clusters"] = final.get("n_clusters")
        metrics["noise_pct"] = round(float(final.get("noise_pct", 0.0)), 1)
        metrics["silhouette"] = round(float(final.get("score", 0.0)), 4)
        for k, v in (opt.get("best_params") or {}).items():
            metrics[f"best_{k}"] = v

    metrics["evidence_pairs"] = count_csv_rows(
        paths["cluster"] / "sam2_cluster_pair_evidence.csv"
    )
    metrics["merge_candidates"] = count_csv_rows(
        paths["cluster"] / "merge_candidates.csv"
    )
    metrics["merged_groups"] = count_csv_rows(paths["merge"] / "merged_groups.csv")

    merged = paths["merge"] / "merged_labels.npy"
    if merged.exists():
        lab = np.load(merged)
        metrics["clusters_after_merge"] = int(lab.max() + 1)

    rows = semantic_table(paths)
    if rows:
        totals = label_totals(rows)
        total = sum(totals.values()) or 1
        for label in ("window", "wall", "roof", "unknown"):
            metrics[f"pct_{label}"] = round(100.0 * totals.get(label, 0) / total, 1)
        metrics["n_labeled_clusters"] = len(rows)

    return metrics


def append_run(cfg: dict, rule: str, exit_code: int, seconds: float,
               log_path: Path) -> dict:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": rule,
        "exit_code": exit_code,
        "seconds": round(seconds, 1),
        "log": log_path.name if log_path else "",
        **config_snapshot(cfg),
        **collect_metrics(cfg),
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    out = []
    for line in HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ----------------------------------------------------------------------

def _report():
    cfg = load_config()
    paths = stage_paths(cfg)
    print(f"repo      : {REPO_ROOT}")
    print(f"dataset   : {paths['dataset_name']}")
    print(f"outputs   : {paths['base']}")
    print()
    for row in stage_status(cfg):
        mark = "done" if row["done"] else " -- "
        print(f"  [{mark}] {row['label']:<24s} {row['mtime']:<17s} {row['target']}")
    print()
    print("metrics:")
    for k, v in collect_metrics(cfg).items():
        print(f"  {k:<26s} {v}")


if __name__ == "__main__":
    _report()
