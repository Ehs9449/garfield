#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize

def colors_for(labels):
    labels=np.asarray(labels,np.int64); rng=np.random.default_rng(12345); u=np.unique(labels)
    pal=rng.integers(35,240,size=(len(u),3),dtype=np.uint8); out=np.zeros((len(labels),3),np.uint8)
    for i,c in enumerate(u): out[labels==c]=pal[i]
    return out

def write_ply(path,xyz,rgb):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); xyz=np.asarray(xyz,np.float32); rgb=np.asarray(rgb,np.uint8)
    dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]); a=np.empty(len(xyz),dtype=dt)
    a['x'],a['y'],a['z']=xyz[:,0],xyz[:,1],xyz[:,2]; a['red'],a['green'],a['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
    hdr=('ply\nformat binary_little_endian 1.0\n'+f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as f: f.write(hdr.encode('ascii')); a.tofile(f)

def main():
    ap=argparse.ArgumentParser(description='Feature-only spherical K-means using the winning K400 settings.')
    ap.add_argument('--points',type=Path,required=True); ap.add_argument('--features',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--k',type=int,default=400); ap.add_argument('--niter',type=int,default=25); ap.add_argument('--nredo',type=int,default=2); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); P=np.load(a.points).astype(np.float32); F=np.load(a.features).astype(np.float32)
    if P.ndim!=2 or P.shape[1]!=3 or F.ndim!=2 or len(P)!=len(F): raise ValueError(f'Invalid arrays: points={P.shape}, features={F.shape}')
    if not np.isfinite(P).all() or not np.isfinite(F).all(): raise ValueError('Input contains NaN/Inf')
    try: import faiss
    except Exception as e: raise RuntimeError('FAISS is required for spherical K-means') from e
    X=np.ascontiguousarray(normalize(F,axis=1).astype(np.float32))
    km=faiss.Kmeans(X.shape[1],a.k,niter=a.niter,nredo=a.nredo,verbose=False,spherical=True,seed=a.seed); km.train(X); sim,L=km.index.search(X,1); labels=L[:,0].astype(np.int32)
    out=a.output_root/'03_spherical_kmeans'; out.mkdir(parents=True,exist_ok=True); stem=f'spherical_kmeans_k{a.k:03d}'
    np.save(out/f'{stem}_labels.npy',labels); write_ply(out/f'{stem}.ply',P,colors_for(labels)); counts=np.bincount(labels,minlength=a.k)
    report={'k':a.k,'n_points':len(P),'n_clusters':int(len(np.unique(labels))),'mean_cosine':float(sim.mean()),'niter':a.niter,'nredo':a.nredo,'seed':a.seed,'median_cluster_size':float(np.median(counts[counts>0])),'largest_cluster_size':int(counts.max())}
    (out/f'{stem}_metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); print('Labels:',out/f'{stem}_labels.npy')
if __name__=='__main__': main()
