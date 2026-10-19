#!/usr/bin/env python3
"""All-class SAM3 matching on viewer-path renders, with sparse z-buffer visibility.

Distinct from the old IoU matcher, this makes direct multi-view point-mask votes
for ALL detected classes, and separately summarizes them per original K400 group.
No retraining, changing K400 labels, or automatic cluster merging.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.ndimage import binary_erosion

COLORS={
    'window':(0,205,255),'wall':(179,179,153),'roof':(204,51,51),
    'door':(204,102,0),'opening':(255,153,0),'column':(230,230,230),
    'beam':(153,76,0),'ceiling':(204,204,230),'floor':(128,128,102),
    'curtain wall':(77,153,204),'ground':(102,102,102),'vegetation':(0,153,0),
    'sky':(128,179,255),'staircase':(255,255,0),'railing':(179,76,179),
    'unknown':(75,75,75),
}

def camera_project(xyz,rec):
    mat=np.asarray(rec['c2w'],np.float64)
    if mat.shape!=(3,4): raise ValueError(f"Bad c2w for {rec['view_name']}: {mat.shape}")
    T=np.eye(4);T[:3]=mat
    C=np.linalg.inv(T)
    cam=xyz @ C[:3,:3].T + C[:3,3]
    d=-cam[:,2]
    idx=np.flatnonzero(d>.01)
    fx,fy,cx,cy=(float(rec[k]) for k in ('fx','fy','cx','cy'))
    u=np.rint(fx*cam[idx,0]/d[idx]+cx).astype(np.int32)
    v=np.rint(-fy*cam[idx,1]/d[idx]+cy).astype(np.int32)
    W,H=int(rec['img_w']),int(rec['img_h'])
    ok=(u>=0)&(u<W)&(v>=0)&(v<H)
    return idx[ok],u[ok],v[ok],d[idx[ok]]

def zbuffer(ids,u,v,d,W,H,tile=4,abs_tol=.02,rel_tol=.025):
    if not len(ids):return ids,u,v
    ntx=(W+tile-1)//tile
    bins=(v//tile)*ntx+(u//tile)
    closest=np.full(ntx*((H+tile-1)//tile),np.inf,np.float32)
    np.minimum.at(closest,bins,d.astype(np.float32))
    near=closest[bins]
    ok=d<=near+np.maximum(abs_tol,rel_tol*near)
    return ids[ok],u[ok],v[ok]

def ply(path,xyz,colors):
    path.parent.mkdir(parents=True,exist_ok=True)
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    vals=np.empty(len(xyz),dtype=dtype)
    for j,key in enumerate(('x','y','z')):vals[key]=xyz[:,j]
    for j,key in enumerate(('red','green','blue')):vals[key]=colors[:,j]
    head=('ply\nformat binary_little_endian 1.0\n'
          f'element vertex {len(xyz)}\n'
          'property float x\nproperty float y\nproperty float z\n'
          'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as f:f.write(head.encode('ascii'));vals.tofile(f)

def class_color(classes):
    rgb=np.empty((len(classes),3),dtype=np.uint8);rgb[:]=COLORS['unknown']
    for c in np.unique(classes):rgb[classes==c]=COLORS.get(str(c),COLORS['unknown'])
    return rgb

def write_csv(path,rows,header):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=header,extrasaction='ignore');wr.writeheader();wr.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    root=Path('outputs/towerlsu')
    ex=root/'feature_clustering_experiments'
    ap.add_argument('--points',type=Path,default=root/'ortho_projection_guided_plus_top/points.npy')
    ap.add_argument('--labels',type=Path,default=ex/'14_kmeans_prototype_merge/k400/merge_cos_0p900/spherical_kmeans_k400_avg_cos_0p900_labels.npy')
    ap.add_argument('--views',type=Path,default=root/'viewer_path_experiment/full/views')
    ap.add_argument('--masks',type=Path,default=root/'viewer_path_experiment/multiclass/masks')
    ap.add_argument('--baseline',type=Path,default=ex/'19_k400_multiclass_evidence/semantic_labels.json')
    ap.add_argument('--previous-pointwise',type=Path,default=ex/'21_k400_spiral_window_fusion/02_point_refined/refined_semantic_labels_per_point.npy')
    ap.add_argument('--output',type=Path,default=ex/'22_viewer_path_full_multiclass')
    ap.add_argument('--min-sam3-score',type=float,default=.30)
    ap.add_argument('--min-evidence-views',type=int,default=2)
    ap.add_argument('--min-point-ratio',type=float,default=.15)
    ap.add_argument('--min-weighted-score',type=float,default=.12)
    ap.add_argument('--window-erode',type=int,default=1)
    ap.add_argument('--window-bonus',type=float,default=1.10)
    ap.add_argument('--tile',type=int,default=4)
    ap.add_argument('--abs-tolerance',type=float,default=.02)
    ap.add_argument('--rel-tolerance',type=float,default=.025)
    ap.add_argument('--allow-missing',action='store_true')
    args=ap.parse_args()
    for f in (args.points,args.labels,args.baseline,args.views/'view_params.json'):
        if not f.is_file():raise FileNotFoundError(f)
    xyz=np.load(args.points).astype(np.float32)
    clusters=np.load(args.labels)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or clusters.shape!=(len(xyz),) or not np.issubdtype(clusters.dtype,np.integer):
        raise ValueError('XYZ or K400 labels invalid')
    n=len(xyz);ngroups=int(clusters.max())+1
    view_params=json.loads((args.views/'view_params.json').read_text())
    if not view_params:raise ValueError('No viewer camera metadata')
    names=[r['view_name'] for r in view_params]
    if len(names)!=len(set(names)):raise ValueError('Duplicate view names')
    bad=[];prompt_sets=[]
    for rec in view_params:
        path=args.masks/(rec['view_name']+'_masks.npz')
        if not path.is_file():bad.append(rec['view_name']);continue
        # Reject stale camera imagery: the mask file must come from this exact JPEG.
        source=args.views/rec.get('image_file',rec['view_name']+'.jpg')
        if not source.is_file():raise FileNotFoundError(source)
        h=hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        with np.load(path,allow_pickle=False) as npz:
            if '_source_sha256' not in npz or str(npz['_source_sha256'].item())!=h.hexdigest():
                raise ValueError(f'{path}: mask/image SHA256 mismatch, rerun SAM3')
            if '_prompts' not in npz:raise ValueError(f'{path} missing _prompts metadata; use viewer multiclass SAM3 script')
            prompt_sets.append(str(npz['_prompts'].item()))
            if int(npz['_width'])!=int(rec['img_w']) or int(npz['_height'])!=int(rec['img_h']):
                raise ValueError(f'{path}: camera/image dimensions differ')
    if bad and not args.allow_missing:raise FileNotFoundError(f'{len(bad)} missing mask files, e.g. {bad[:5]}')
    if not prompt_sets:raise RuntimeError('No multiclass masks')
    if len(set(prompt_sets))!=1:raise ValueError('Mixed SAM3 prompt sets in same mask directory')
    classes=prompt_sets[0].split('|')
    if not classes or classes==['window']:raise ValueError('Option B requires all-class SAM3 masks, not just window')
    if not 0 <= args.window_erode <= 5: raise ValueError('window-erode outside 0..5')
    cls_num=len(classes)
    counts=np.zeros((cls_num,n),np.uint16)
    confidence=np.zeros((cls_num,n),np.float32)
    opportunities=np.zeros(n,np.uint16)
    per_view=[]
    print(f'All-class independent matching: {n:,} points, {ngroups} K400 groups, {len(view_params)} cameras, {len(classes)} classes')
    print('Classes:',classes)
    for k,r in enumerate(view_params,1):
        name=r['view_name'];file=args.masks/(name+'_masks.npz')
        if not file.exists():continue
        W,H=int(r['img_w']),int(r['img_h'])
        idx,u,v,d=camera_project(xyz,r)
        n_projected=len(idx)
        idx,u,v=zbuffer(idx,u,v,d,W,H,args.tile,args.abs_tolerance,args.rel_tolerance)
        opportunities[idx]+=1
        hits_per_class={};kept_masks={}
        with np.load(file,allow_pickle=False) as data:
            for ci,cl in enumerate(classes):
                key=cl.replace(' ','_')
                if key+'_masks' not in data:raise ValueError(f'{file}: missing {key}_masks')
                mats=data[key+'_masks'];scores=data[key+'_scores']
                if mats.ndim!=3 or mats.shape[1:]!=(H,W):raise ValueError(f'{file}: bad {key}_masks shape {mats.shape}')
                if len(mats)!=len(scores):raise ValueError(f'{file}: bad {key} scores')
                mask_conf=np.zeros(len(idx),np.float32)
                keep=0
                for mask,score in zip(mats,scores):
                    score=float(score)
                    if score<args.min_sam3_score or not mask.any():continue
                    keep+=1
                    if cl=='window' and args.window_erode:
                        mask=binary_erosion(mask.astype(bool),iterations=args.window_erode)
                    if len(idx):
                        selected=mask[v,u]
                        np.maximum(mask_conf,selected.astype(np.float32)*score,out=mask_conf)
                supported=mask_conf>0
                counts[ci,idx[supported]]+=1
                confidence[ci,idx]+=mask_conf
                hits_per_class[cl]=int(supported.sum())
                kept_masks[cl]=keep
        per_view.append({'view':name,'projected_points':n_projected,'zbuffer_visible':len(idx),
                         'n_detected_masks':sum(kept_masks.values()),'window_hits':hits_per_class.get('window',0),
                         'class_mask_counts':json.dumps(kept_masks,sort_keys=True),
                         'class_point_hits':json.dumps(hits_per_class,sort_keys=True)})
        print(f'[{k}/{len(view_params)}] {name}: visible {len(idx):,}; masks {sum(kept_masks.values())}; window hits {hits_per_class.get("window",0):,}',flush=True)
    visible=np.maximum(opportunities,1)
    hit_fraction=counts.astype(np.float32)/visible[None,:]
    vote_score=confidence/visible[None,:]
    eligible=(counts>=args.min_evidence_views)&(hit_fraction>=args.min_point_ratio)&(vote_score>=args.min_weighted_score)
    score=vote_score.copy()
    score[~eligible]=-1
    if 'window' in classes:
        score[classes.index('window'),eligible[classes.index('window')]]*=args.window_bonus
    winners=np.argmax(score,axis=0)
    best=score[winners,np.arange(n)]
    new_semantic=np.full(n,'unknown',dtype='<U32')
    supported=best>=0
    class_array=np.asarray(classes,dtype='<U32')
    new_semantic[supported]=class_array[winners[supported]]
    with args.baseline.open() as f:baseline=json.load(f)
    baseline_cluster=np.asarray([str(baseline.get(str(g),{}).get('label','unknown')).lower() for g in range(ngroups)],dtype='<U32')
    base_sem=np.full(n,'unknown',dtype='<U32')
    valid=(clusters>=0)
    base_sem[valid]=baseline_cluster[clusters[valid]]
    refined=new_semantic.copy();refined[~supported]=base_sem[~supported]
    # K400 cluster-level aggregation (separate diagnostic), never relabel full clusters in pointwise output.
    group_sem=baseline_cluster.copy()
    cluster_rows=[]
    nonnegative=clusters>=0
    for group in range(ngroups):
        member=clusters==group;tot=int(member.sum())
        if not tot:continue
        score_totals=[]
        support_counts=[]
        for ci,cl in enumerate(classes):
            validpts=member & eligible[ci]
            score_totals.append(float(vote_score[ci,validpts].sum()))
            support_counts.append(int(validpts.sum()))
        leader=int(np.argmax(score_totals))
        leader_support=int(support_counts[leader])
        leader_label=classes[leader]
        if leader_support>=max(4,args.min_evidence_views) and score_totals[leader]>0:
            group_sem[group]=leader_label
        cluster_rows.append(dict(cluster_id=group,n_points=tot,baseline_label=baseline_cluster[group],
                                 viewer_label=group_sem[group],supported_points=int(np.count_nonzero(member & supported)),
                                 dominant_class=leader_label,dominant_class_supported_points=leader_support,
                                 class_score_sums=json.dumps(dict(zip(classes,[round(q,4) for q in score_totals]))),
                                 class_supported_points=json.dumps(dict(zip(classes,support_counts)))))
    group_pts=np.full(n,'unknown',dtype='<U32')
    group_pts[nonnegative]=group_sem[clusters[nonnegative]]
    out=args.output.resolve()
    for inp in (args.points,args.labels,args.views,args.masks,args.baseline):
        if out==inp.resolve() or out.is_relative_to(inp.resolve()):raise ValueError('Output cannot overwrite an input path')
    out.mkdir(parents=True,exist_ok=True)
    sem=out/'semantic';diag=out/'diagnostics';sem.mkdir(exist_ok=True);diag.mkdir(exist_ok=True)
    ply(sem/'semantic_pointcloud_multiclass_pointwise.ply',xyz,class_color(refined))
    ply(sem/'semantic_pointcloud_new_evidence_only.ply',xyz,class_color(new_semantic))
    ply(sem/'semantic_pointcloud_multiclass_clusterwise.ply',xyz,class_color(group_pts))
    np.save(sem/'semantic_labels_pointwise.npy',refined)
    np.save(sem/'semantic_labels_evidence_only.npy',new_semantic)
    np.save(sem/'window_supported_mask.npy', (new_semantic=='window'))
    label_ply_dir=sem/'per_label';label_ply_dir.mkdir(exist_ok=True)
    for cl in set(refined.tolist()):
        selected=refined==cl
        if selected.any():ply(label_ply_dir/(cl.replace(' ','_')+'.ply'),xyz[selected],class_color(refined[selected]))
    write_csv(diag/'per_view_class_matching.csv',per_view,
              ['view','projected_points','zbuffer_visible','n_detected_masks','window_hits','class_mask_counts','class_point_hits'])
    write_csv(diag/'per_cluster_class_evidence.csv',cluster_rows,
              ['cluster_id','n_points','baseline_label','viewer_label','supported_points','dominant_class',
               'dominant_class_supported_points','class_score_sums','class_supported_points'])
    all_classes=sorted(set(COLORS)|set(classes)|set(base_sem.tolist())|set(refined.tolist()))
    previous=None
    if args.previous_pointwise.is_file():
        previous=np.load(args.previous_pointwise,allow_pickle=False)
        if previous.shape!=(n,):previous=None
    summary_rows=[]
    for cl in all_classes:
        summary_rows.append(dict(semantic_class=cl,global_baseline=int((base_sem==cl).sum()),
                                 previous_spiral=int((previous==cl).sum()) if previous is not None else 'missing',
                                 viewer_evidence_only=int((new_semantic==cl).sum()),
                                 viewer_with_fallback=int((refined==cl).sum()),
                                 viewer_clusterwise=int((group_pts==cl).sum())))
    write_csv(diag/'comparison_vs_previous.csv',summary_rows,
              ['semantic_class','global_baseline','previous_spiral','viewer_evidence_only',
               'viewer_with_fallback','viewer_clusterwise'])
    summary=dict(total_points=n,clusters=ngroups,views=len(view_params),views_with_masks=len(view_params)-len(bad),
                 classes=classes,points_with_new_semantic_evidence=int(supported.sum()),
                 points_falling_back_to_global_baseline=int((~supported).sum()),
                 baseline_counts=dict(Counter(base_sem.tolist())),
                 new_evidence_only_counts=dict(Counter(new_semantic.tolist())),
                 final_pointwise_counts=dict(Counter(refined.tolist())),
                 thresholds=dict(min_sam3_score=args.min_sam3_score,min_evidence_views=args.min_evidence_views,
                    min_point_ratio=args.min_point_ratio,min_weighted_score=args.min_weighted_score,
                    window_erode=args.window_erode,window_bonus=args.window_bonus,
                    zbuffer_tile=args.tile,depth_abs=args.abs_tolerance,depth_rel=args.rel_tolerance),
                 important_limitations=['Approximate z-buffer from point-cloud points; not rasterized Gaussian depth',
                    'Multi-class SAM3 masks may overlap and are not ground truth',
                    'Fallback baseline labels are original global-view semantic categories; no new evidence is labeled unknown in evidence-only PLY',
                    'Clusters are unchanged, clusterwise PLY is only a diagnostic; pointwise PLY does NOT promote entire K400 cluster'])
    (diag/'summary.json').write_text(json.dumps(summary,indent=2))
    (sem/'cluster_semantic_labels.json').write_text(json.dumps({str(i):{'label':str(group_sem[i]),'points':int((clusters==i).sum())}
                                                        for i in range(ngroups)},indent=2))
    print('\n=== Completed full viewer-path semantic matching ===')
    print('Evidence-supported points:',int(supported.sum()),'of',n)
    for row in summary_rows:
        if row['viewer_with_fallback'] or row['global_baseline']:
            print(f" {row['semantic_class']:14s} global={row['global_baseline']:7d} previous={str(row['previous_spiral']):>7s} viewer={row['viewer_with_fallback']:7d}")
    print('CloudCompare (main):',sem/'semantic_pointcloud_multiclass_pointwise.ply')
    print('Clusterwise diagnostic:',sem/'semantic_pointcloud_multiclass_clusterwise.ply')
    print('Comparison:',diag/'comparison_vs_previous.csv')

if __name__=='__main__':main()
