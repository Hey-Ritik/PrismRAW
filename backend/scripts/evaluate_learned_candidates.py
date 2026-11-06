"""Compare isolated experiments on development validation, never select on test."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from smartphone_isp.evaluation.metrics import quality
from smartphone_isp.pipelines.learned import load_model, reconstruct
from smartphone_isp.pipelines.classical import render_linear

torch.set_num_threads(2)
manifest_path = Path('data/processed/final/manifest.json')
manifest = json.loads(manifest_path.read_text())
rows = [row for row in manifest['samples'] if row['split']=='validation']
names = ['final-learned', 'learned-detail-candidate', 'learned-detail-refine', 'learned-detail-expanded']
report = {'split':'validation','scene_ids':sorted({r['scene_id'] for r in rows}),'patch_count':len(rows),
          'selection_rule':'Require lower LPIPS, no decrease in PSNR/SSIM and no increase in Delta E versus deployed baseline; choose lowest LPIPS among eligible candidates.',
          'test_used_for_selection':False,'experiments':{}}
for name in names:
    path = Path('checkpoints')/f'{name}.pt'
    model, document = load_model(path)
    scores = []
    for row in rows:
        with np.load(manifest_path.parent/row['path'],allow_pickle=False) as pair:
            linear = reconstruct(model,pair['packed_raw'],pair['calibration'])
            image = render_linear(linear,float(pair['exposure']),float(pair['sharpen']))[0]
            c = manifest['input_context']
            if c:image=image[c:-c,c:-c]
            scores.append(quality(image,pair['target_srgb']))
    metrics = {key:float(np.mean([m[key] for m in scores])) for key in scores[0]}
    training = Path(document['config']['output'])/'training.json'
    log = json.loads(training.read_text())
    report['experiments'][name]={'metrics':metrics,'best_validation_linear_l1':document['validation_l1'],
                               'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'config':document['config'],'epochs_run':len(log['epochs']),
                               'train_samples':log['train_samples'],'validation_samples':log['validation_samples']}
    print(name,metrics,flush=True)
    del model
baseline = report['experiments']['final-learned']['metrics']
eligible=[]
for name,experiment in report['experiments'].items():
    m=experiment['metrics']
    accepted = name!='final-learned' and m['lpips']<baseline['lpips'] and m['psnr']>=baseline['psnr'] and m['ssim']>=baseline['ssim'] and m['delta_e_00']<=baseline['delta_e_00']
    experiment['eligible']=accepted
    if accepted:eligible.append(name)
report['selected']=min(eligible,key=lambda name:report['experiments'][name]['metrics']['lpips']) if eligible else 'final-learned'
report['production_weights_replaced']=False
Path('reports/learned_improvement_trials.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Validation selected:',report['selected'],'; no weights replaced by this script.',flush=True)
