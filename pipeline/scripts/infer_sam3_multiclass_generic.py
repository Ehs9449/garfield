#!/usr/bin/env python3
"""Fine-tuned SAM3 on viewer-path JPGs; isolated HPC output; resumable per image."""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config-yaml',required=True)
    p.add_argument('--views-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prompts',nargs='+',required=True)
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    jpgs=sorted(a.views_dir.glob('*.jpg'))
    if not jpgs: raise FileNotFoundError('No *.jpg files in '+str(a.views_dir))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    import torch
    import torch.nn.functional as F
    import yaml
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    cfg=yaml.safe_load(Path(a.config_yaml).read_text())
    model=build_sam3_image_model(bpe_path=cfg['sam3']['bpe_path'],load_from_HF=False)
    ckpt=torch.load(cfg['sam3']['checkpoint'],map_location='cpu',weights_only=False)
    state=ckpt['model'] if isinstance(ckpt,dict) and 'model' in ckpt else ckpt
    status=model.load_state_dict(state,strict=False)
    print('Checkpoint:',cfg['sam3']['checkpoint'],'missing',len(status.missing_keys),'unexpected',len(status.unexpected_keys),flush=True)
    model=model.cuda().eval()
    processor=Sam3Processor(model,confidence_threshold=float(cfg['sam3'].get('confidence_threshold',0.3)))
    prompt_keys=[s.replace(' ','_') for s in a.prompts]
    if len(set(prompt_keys)) != len(prompt_keys): raise ValueError('Duplicate prompts')
    rows=[]
    for i, imgp in enumerate(jpgs,1):
        dest=a.output_dir/(imgp.stem+'_masks.npz')
        sha=digest(imgp)
        if dest.exists() and not a.force:
            try:
                with np.load(dest,allow_pickle=False) as old:
                    cached_keys=str(old['_prompts'].item()) if '_prompts' in old else ''
                    cached_sha=str(old['_source_sha256'].item()) if '_source_sha256' in old else ''
                    if cached_keys == '|'.join(a.prompts) and cached_sha==sha:
                        print(f'[{i}/{len(jpgs)}] cache hit {imgp.name}',flush=True)
                        continue
            except Exception as exc: print('Ignoring invalid cache:',exc,flush=True)
        img=Image.open(imgp).convert('RGB')
        w,h=img.size
        payload={'_prompts':np.array('|'.join(a.prompts)), '_source_sha256':np.array(sha),
                 '_width':np.array(w), '_height':np.array(h)}
        print(f'[{i}/{len(jpgs)}] {imgp.name} {w}x{h}',flush=True)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            for prompt,key in zip(a.prompts,prompt_keys):
                result=processor.set_image(img)
                result=processor.set_text_prompt(state=result,prompt=prompt)
                masks=result.get('masks')
                if masks is None or masks.numel()==0:
                    payload[key+'_masks']=np.zeros((0,h,w),dtype=bool)
                    payload[key+'_scores']=np.empty((0,),dtype=np.float32)
                    payload[key+'_boxes']=np.empty((0,4),dtype=np.float32)
                    count=0
                else:
                    if masks.ndim==3: masks=masks.unsqueeze(1)
                    if masks.ndim!=4 or masks.shape[1]!=1:
                        raise ValueError(f'Unexpected mask shape: {tuple(masks.shape)}')
                    if masks.shape[-2:]!=(h,w):
                        print(f'  {prompt}: resizing masks {tuple(masks.shape[-2:])} to {(h,w)}',flush=True)
                        masks=F.interpolate(masks.float(),size=(h,w),mode='nearest')
                    data=masks[:,0].detach().cpu().numpy().astype(bool)
                    sc=result['scores'].detach().float().cpu().numpy().reshape(-1)
                    box=result['boxes'].detach().float().cpu().numpy().reshape(-1,4)
                    if not (len(data)==len(sc)==len(box)): raise ValueError('SAM3 mask/score/box size mismatch')
                    payload[key+'_masks']=data
                    payload[key+'_scores']=sc.astype(np.float32)
                    payload[key+'_boxes']=box.astype(np.float32)
                    count=len(data)
                rows.append(dict(view=imgp.stem,semantic_class=prompt,num_masks=count,width=w,height=h))
                print(' ',prompt,count,flush=True)
        tmp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**payload)
        os.replace(tmp,dest)
    with (a.output_dir/'mask_report.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['view','semantic_class','num_masks','width','height'])
        writer.writeheader();writer.writerows(rows)
    print('Saved independent multi-class masks in',a.output_dir,flush=True)

if __name__=='__main__': main()
