#!/usr/bin/env python3
"""Ensure every rendered image has fresh, correctly aligned SAM3 masks."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify(views,masks,prompts,quiet=False):
    meta=json.loads((views/'view_params.json').read_text())
    bad=[]
    if not meta:raise ValueError('Empty view_params.json')
    for record in meta:
        name=record['view_name'];jpg=views/record.get('image_file',name+'.jpg')
        npz=masks/(name+'_masks.npz')
        if not jpg.is_file():bad.append(name+': missing rendered image');continue
        if not npz.is_file():bad.append(name+': missing .npz');continue
        try:
            with np.load(npz,allow_pickle=False) as obj:
                if str(obj['_prompts'].item())!='|'.join(prompts):
                    bad.append(name+': mismatched prompt set');continue
                if str(obj['_source_sha256'].item())!=sha256(jpg):
                    bad.append(name+': stale (image hash changed)');continue
                h=int(record['img_h']);w=int(record['img_w'])
                for cl in prompts:
                    key=cl.replace(' ','_')+'_masks'
                    if key not in obj or obj[key].ndim!=3 or tuple(obj[key].shape[1:])!=(h,w):
                        bad.append(name+': invalid '+key);break
        except Exception as e:bad.append(name+': '+repr(e))
    if not quiet:
        print(f'Validated {len(meta)-len(bad)}/{len(meta)} viewer SAM3 mask files')
        for s in bad[:12]:print('  ',s)
    return not bad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--views',type=Path,required=True);ap.add_argument('--masks',type=Path,required=True)
    ap.add_argument('--prompts',nargs='+',required=True)
    a=ap.parse_args()
    sys.exit(0 if verify(a.views,a.masks,a.prompts) else 1)
if __name__=='__main__': main()
