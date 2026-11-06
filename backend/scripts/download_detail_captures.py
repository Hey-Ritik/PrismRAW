import json,sys,urllib.request,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
from download_sidd_subset import request_json,sha256,REPO,REVISION
index=json.loads(Path('data/raw/sidd-cc-subset/pilot-index.json').read_text())
for directory in ['Raw/train/Data/0064_003_GP_01600_01600_4400_N','Raw/train/Data/0076_004_N6_03200_00320_3200_L']:
 entries=request_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{directory}')
 byname={e['path'].split('/')[-1]:e for e in entries}
 noisy=sorted(n for n in byname if '_NOISY_RAW_' in n)[0]
 record={'scene_id':directory.split('/')[-1].split('_')[1],'instance':directory.split('/')[-1],'files':{}}
 for kind,name in zip(['noisy','clean','metadata'],[noisy,noisy.replace('_NOISY_','_GT_'),noisy.replace('_NOISY_','_METADATA_')]):
  info=byname[name]; target=Path('data/raw/sidd-cc-subset')/record['instance']/name; target.parent.mkdir(parents=True,exist_ok=True)
  url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{info["path"]}'
  expected=info.get('lfs',{}).get('oid')
  if not target.exists() or target.stat().st_size!=info['size']:
   print('Downloading',name,round(info['size']/1e6,1),'MB',flush=True)
   temporary=target.with_suffix('.part')
   with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as out:
    while chunk:=response.read(1024*1024):out.write(chunk)
   assert temporary.stat().st_size==info['size']
   if expected: assert sha256(temporary)==expected
   temporary.replace(target)
  digest=sha256(target)
  if expected:assert digest==expected
  record['files'][kind]={'path':target.as_posix(),'url':url,'sha256':digest,'bytes':target.stat().st_size}
 index['records'].append(record)
Path('data/raw/sidd-cc-subset/detail-index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
print('Broader training captures verified; original index preserved.',flush=True)
