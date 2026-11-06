"""Document measured weaknesses with reproducible held-out visual examples."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from smartphone_isp.pipelines.learned import load_model, reconstruct
from smartphone_isp.pipelines.hybrid import load_model as load_hybrid, refine
from smartphone_isp.pipelines.classical import render_linear

torch.set_num_threads(2)
root = Path.cwd()
results = json.loads(Path('reports/final_results.json').read_text())
manifest = json.loads(Path('data/processed/final/manifest.json').read_text())
rows = {row['sample_id']: row for row in manifest['samples']}
learned = load_model('checkpoints/final-learned.pt')[0]
hybrid = load_hybrid('checkpoints/final-hybrid.pt')[0]
examples = [
    ('learned_perceptual_error', max(results['samples'], key=lambda r:r['metrics']['learned']['lpips'])),
    ('hybrid_tradeoff', max(results['samples'], key=lambda r:r['metrics']['hybrid']['lpips']-r['metrics']['classical']['lpips']))
]
destination = Path('reports/failure_examples'); destination.mkdir(exist_ok=True)
cases = []
for name, sample in examples:
    row = rows[sample['sample_id']]
    with np.load(Path('data/processed/final')/row['path'], allow_pickle=False) as pair:
        context = manifest['input_context']
        def display(linear):
            value = render_linear(linear, float(pair['exposure']), float(pair['sharpen']))[0]
            return value[context:-context, context:-context] if context else value
        images = {'Reference':pair['target_srgb'], 'Classical':pair['classical'],
                  'Learned':display(reconstruct(learned,pair['packed_raw'],pair['calibration'])),
                  'Hybrid':display(refine(hybrid,pair['denoised'],*pair['conditions']))}
        sheet = Image.new('RGB',(4*256,288),'#f5f7f0'); draw = ImageDraw.Draw(sheet)
        for i,(label,array) in enumerate(images.items()):
            image = Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8))
            sheet.paste(image.resize((256,256),Image.Resampling.NEAREST),(256*i,32))
            draw.text((256*i+10,8),label,fill='#15251f')
        sheet.save(destination/f'{name}.png')
    m = sample['metrics']
    explanation = (f"Learned LPIPS {m['learned']['lpips']:.4f}, versus classical {m['classical']['lpips']:.4f}; learned colour error {m['learned']['delta_e_00']:.2f}, versus classical {m['classical']['delta_e_00']:.2f}." if name.startswith('learned') else
                   f"Hybrid LPIPS {m['hybrid']['lpips']:.4f}, versus classical {m['classical']['lpips']:.4f}; hybrid PSNR {m['hybrid']['psnr']:.2f} dB, versus classical {m['classical']['psnr']:.2f} dB. Metrics capture different trade-offs.")
    cases.append({'name':name,'sample_id':sample['sample_id'],'scene_id':sample['scene_id'],
                  'image':f'/failure-examples/{name}.png','metrics':m,'explanation':explanation})
report = {'scope':'Retrospectively selected difficult examples from the frozen 64-patch test report; not a new independent evaluation or model-selection set.',
          'examples':cases,'checkpoint_sha256':{name:hashlib.sha256(Path(f'checkpoints/final-{name}.pt').read_bytes()).hexdigest() for name in ('learned','hybrid')},
          'limitations':['Two physical test scenes cannot establish broad camera generalization.', 'Clean RAW targets use approximate colour calibration.',
                         'Synthetic dark/noisy and colour-cast stress results are distinct from real capture evaluation.',
                         'LPIPS and pixel fidelity can disagree; inspect images alongside multiple metrics.']}
Path('reports/failure_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
Path('reports/failure_analysis.md').write_text('# Failure analysis\n\n'+report['scope']+'\n\n'+'\n\n'.join(f"## {c['name']} — {c['sample_id']} (scene {c['scene_id']})\n\n{c['explanation']}\n\n![Reference, classical, learned, hybrid](failure_examples/{c['name']}.png)" for c in cases)+'\n\n## Limits\n\n'+'\n'.join('- '+s for s in report['limitations'])+'\n',encoding='utf-8')
trials_path = Path('reports/learned_improvement_trials.json')
if trials_path.exists():
    trials = json.loads(trials_path.read_text(encoding='utf-8'))
    report['training_followup'] = {'selected':trials['selected'], 'selection_rule':trials['selection_rule'],
                                   'production_weights_replaced':trials['production_weights_replaced']}
    Path('reports/failure_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with Path('reports/failure_analysis.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## Learned-model follow-up\n\nThree isolated training experiments were evaluated on the same 64 development-validation patches. Test patches were not used for selection. No candidate passed the combined PSNR, SSIM, colour-error and LPIPS gate; production weights were retained.\n\n')
        stream.write('| Candidate | Validation PSNR | SSIM | Delta E | LPIPS |\n|---|---:|---:|---:|---:|\n')
        for name,trial in trials['experiments'].items():
            m=trial['metrics']
            stream.write(f"| {name} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['delta_e_00']:.4f} | {m['lpips']:.4f} |\n")
        stream.write('\nThe expanded trial adds two high-ISO captures from training scenes 003/004, increasing training patches from 192 to 256 while preserving all scene assignments. Higher PSNR alone did not establish better overall reconstruction. Full measured results and configurations are in learned_improvement_trials.json.\n')
print('Wrote measured failure examples and analysis.')
