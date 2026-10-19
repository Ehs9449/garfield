#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
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

ARCH_CLASSES = ('wall','roof','curtain wall','opening','beam','column','door','window')
EROSION = {'wall':2,'roof':2,'curtain wall':2,'opening':2,'beam':1,'column':1,'door':2,'window':2}

def project_points(xyz, rec):
    c2w=np.asarray(rec['c2w'],np.float64)
    T=np.eye(4); T[:3]=c2w
    w2c=np.linalg.inv(T)
    cam=xyz@w2c[:3,:3].T+w2c[:3,3]
    depth=-cam[:,2]
    ids=np.flatnonzero(depth>0.01)
    fx,fy,cx,cy=(float(rec[k]) for k in ('fx','fy','cx','cy'))
    u=np.rint(fx*cam[ids,0]/depth[ids]+cx).astype(np.int32)
    v=np.rint(-fy*cam[ids,1]/depth[ids]+cy).astype(np.int32)
    W,H=int(rec['img_w']),int(rec['img_h'])
    ok=(u>=0)&(u<W)&(v>=0)&(v<H)
    return ids[ok],u[ok],v[ok],depth[ids[ok]]

def zbuffer(ids,u,v,depth,W,H,tile=4,abs_tol=.02,rel_tol=.025):
    if len(ids)==0: return ids,u,v
    nx=(W+tile-1)//tile
    bid=(v//tile)*nx+(u//tile)
    closest=np.full(nx*((H+tile-1)//tile),np.inf,np.float32)
    np.minimum.at(closest,bid,depth.astype(np.float32))
    near=closest[bid]
    keep=depth<=near+np.maximum(abs_tol,rel_tol*near)
    return ids[keep],u[keep],v[keep]

def write_ply(path,xyz,labels):
    path.parent.mkdir(parents=True,exist_ok=True)
    rgb=np.empty((len(labels),3),np.uint8); rgb[:]=COLORS['unknown']
    for c in np.unique(labels):
        rgb[labels==c]=COLORS.get(str(c),COLORS['unknown'])
    arr=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate('xyz'): arr[k]=xyz[:,i]
    for i,k in enumerate(('red','green','blue')): arr[k]=rgb[:,i]
    head=('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(xyz)}\n'
          'property float x\nproperty float y\nproperty float z\n'
          'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as f:
        f.write(head.encode()); arr.tofile(f)

def run(a):
    for p in (a.points,a.semantic_labels,a.views/'view_params.json'):
        if not p.is_file(): raise FileNotFoundError(p)
    if not a.masks.is_dir(): raise FileNotFoundError(a.masks)

    xyz=np.load(a.points).astype(np.float32)
    labels=np.load(a.semantic_labels,allow_pickle=False).astype('<U32')
    records=json.loads((a.views/'view_params.json').read_text())
    sky_ids=np.flatnonzero(labels=='sky')
    print(f'Total points: {len(xyz):,}')
    print(f'Current sky points: {len(sky_ids):,}')

    out=a.output; out.mkdir(parents=True,exist_ok=True)
    if len(sky_ids)==0:
        np.save(out/'semantic_labels.npy',labels)
        write_ply(out/'semantic_pointcloud.ply',xyz,labels)
        print('No sky points; nothing to change.')
        return

    sky_xyz=xyz[sky_ids]; n=len(sky_xyz)
    hits={c:np.zeros(n,np.uint16) for c in ARCH_CLASSES}
    opp={c:np.zeros(n,np.uint16) for c in ARCH_CLASSES}
    confsum={c:np.zeros(n,np.float32) for c in ARCH_CLASSES}
    rows=[]; views_with_masks=Counter()

    for i,rec in enumerate(records,1):
        name=rec['view_name']
        mf=a.masks/f'{name}_masks.npz'
        rgb=a.views/rec.get('image_file',name+'.jpg')
        if not mf.is_file() or not rgb.is_file():
            raise FileNotFoundError(f'Missing mask/image for {name}')
        W,H=int(rec['img_w']),int(rec['img_h'])
        maps={}; kept={}
        with np.load(mf,allow_pickle=False) as data:
            if '_source_sha256' in data:
                got=str(data['_source_sha256'].item())
                want=hashlib.sha256(rgb.read_bytes()).hexdigest()
                if got!=want: raise ValueError(f'{name}: stale masks')
            for c in ARCH_CLASSES:
                key=c.replace(' ','_'); mk=key+'_masks'; sk=key+'_scores'
                if mk not in data or sk not in data:
                    raise ValueError(f'{mf} missing {c} masks/scores')
                smap=np.zeros((H,W),np.float32); kc=0
                for m,s in zip(data[mk],data[sk]):
                    s=float(s)
                    if s<a.min_sam3_score or not m.any(): continue
                    mm=m.astype(bool)
                    if EROSION[c]>0: mm=binary_erosion(mm,iterations=EROSION[c])
                    if not mm.any(): continue
                    kc+=1; smap[mm]=np.maximum(smap[mm],s)
                maps[c]=smap; kept[c]=kc
        active=[c for c in ARCH_CLASSES if kept[c]>0]
        visible=0; class_hits={c:0 for c in ARCH_CLASSES}
        if active:
            idx,u,v,d=project_points(sky_xyz,rec)
            idx,u,v=zbuffer(idx,u,v,d,W,H,a.tile,a.abs_tol,a.rel_tol)
            visible=len(idx)
            for c in active:
                views_with_masks[c]+=1
                opp[c][idx]+=1
                ps=maps[c][v,u]
                matched=ps>0
                local_ids=idx[matched]
                hits[c][local_ids]+=1
                confsum[c][local_ids]+=ps[matched]
                class_hits[c]=len(local_ids)
        row={'view':name,'visible_sky_points':visible}
        for c in ARCH_CLASSES:
            row[f'{c}_masks']=kept[c]
            row[f'{c}_sky_hits']=class_hits[c]
        rows.append(row)
        print(f'[{i}/{len(records)}] {name}: visible_sky={visible:,}',flush=True)

    evidence={}
    for c in ARCH_CLASSES:
        ratio=hits[c].astype(np.float32)/np.maximum(opp[c],1)
        mean_conf=np.divide(confsum[c],np.maximum(hits[c],1),dtype=np.float32)
        ev=ratio*mean_conf
        ev[hits[c]==0]=-1.0
        evidence[c]=ev

    stack=np.stack([evidence[c]+rank*1e-7 for rank,c in enumerate(ARCH_CLASSES)],axis=0)
    winner=np.argmax(stack,axis=0)
    best=np.max(stack,axis=0)
    has_arch=best>=0

    refined=labels.copy(); corrected=Counter(); changed=[]
    for idx_c,c in enumerate(ARCH_CLASSES):
        local=has_arch&(winner==idx_c)
        gids=sky_ids[local]
        refined[gids]=c
        corrected[c]=int(len(gids))
        if len(gids): changed.append(gids)
    changed=np.concatenate(changed) if changed else np.zeros(0,dtype=np.int64)

    np.save(out/'semantic_labels.npy',refined)
    np.save(out/'corrected_from_sky_point_ids.npy',changed)
    write_ply(out/'semantic_pointcloud.ply',xyz,refined)

    per=out/'per_class'; per.mkdir(exist_ok=True)
    counts=Counter(refined.tolist())
    for c in sorted(counts):
        sel=refined==c
        write_ply(per/f'{c.replace(" ","_")}.ply',xyz[sel],refined[sel])

    if len(changed):
        write_ply(out/'sky_corrected_points_only.ply',xyz[changed],refined[changed])
    remaining=refined=='sky'
    if remaining.any():
        write_ply(out/'remaining_sky_only.ply',xyz[remaining],refined[remaining])

    with (out/'per_view_sky_architecture_evidence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary={
        'total_points':int(len(xyz)),
        'initial_sky_points':int(len(sky_ids)),
        'corrected_sky_points':int(len(changed)),
        'remaining_sky_points':int(remaining.sum()),
        'corrected_to':dict(corrected),
        'final_class_counts':dict(counts),
        'rule':'architecture always beats sky when at least one valid architectural SAM3 hit exists',
        'all_non_sky_labels_preserved_exactly':True,
        'min_sam3_score':float(a.min_sam3_score),
        'classes':list(ARCH_CLASSES),
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('\n=== DONE: ARCHITECTURE > SKY ===')
    print(json.dumps(summary,indent=2))
    print('Semantic PLY:',out/'semantic_pointcloud.ply')

def main():
    p=argparse.ArgumentParser()
    root=Path('outputs/towerlsu'); exp=root/'feature_clustering_experiments'
    p.add_argument('--points',type=Path,default=root/'ortho_projection_guided_plus_top/points.npy')
    p.add_argument('--semantic-labels',type=Path,default=exp/'25_viewer_component_aware/viewer_only_strict/semantic_labels.npy')
    p.add_argument('--views',type=Path,default=root/'viewer_path_experiment/full/views')
    p.add_argument('--masks',type=Path,default=root/'viewer_path_experiment/multiclass/masks')
    p.add_argument('--output',type=Path,default=exp/'27_component_aware_architecture_over_sky/viewer_only_strict')
    p.add_argument('--min-sam3-score',type=float,default=.30)
    p.add_argument('--tile',type=int,default=4)
    p.add_argument('--abs-tol',type=float,default=.02)
    p.add_argument('--rel-tol',type=float,default=.025)
    run(p.parse_args())

if __name__=='__main__':
    main()
