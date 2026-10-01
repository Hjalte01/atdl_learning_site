"""Reproduce V-JEPA 2.1 v3 teaching SVGs; run from repository root."""
from pathlib import Path
from html import escape
from math import isclose, sqrt
out=Path('public/figures/vjepa21');out.mkdir(parents=True,exist_ok=True)
def diagram(name,title,rows):
 h=100+len(rows)*145
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{h}" viewBox="0 0 960 {h}" role="img" aria-label="{escape(title)}">','<rect width="100%" height="100%" fill="#101c30"/>','<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="24">{escape(title)}</text>']
 for i,(label,boxes) in enumerate(rows):
  y=80+i*145;p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
  for j,lines in enumerate(boxes):
   x=30+j*310;p.append(f'<rect x="{x}" y="{y+15}" width="275" height="90" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
   for k,line in enumerate(lines):p.append(f'<text x="{x+12}" y="{y+40+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
   if j<len(boxes)-1:p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
 p.append('</svg>');(out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','Recognizing a grasp does not locate the cup boundary',[
('MASKED-ONLY OBJECTIVE: visible tokens help predict elsewhere', [('Clip: hand reaches for cup','Hide some patch positions'),('Visible features → predictor','Loss only at hidden sites'),('Can aggregate global cues','Local structure harder to read')]),
('DENSE OBJECTIVE: visible outputs also have location-specific targets', [('Same clip, same masking','Clean teacher: all patches'),('Predict hidden + visible sites','Compare each to its target'),('Encourage local grounding','Still learn scene dynamics')])])
diagram('architecture','A · tokenize → B · encode → C · predict → D · compare',[
('STUDENT PATH: gradient updates through B and C', [('A: 2D image / 3D video conv','Drop masked patch tokens'),('B: visible-token ViT','4 normalized layer outputs'),('Concat channels → MLP','Fuse and reduce channels')]),
('CONTINUATION: fused context enters C', [('Fused context at visible sites','Add positional mask tokens'),('C: 24-block predictor','Width 384 during pretraining'),('4 output feature levels','For visible + masked sites')]),
('TEACHER PATH: targets for D; no backpropagation', [('A: same clean image / clip','Keep all patch positions'),('EMA y-encoder','4 corresponding layer outputs'),('D: masked L1 + context L1','Stop gradient on targets')]),
('UPDATE LOOP: loss gradients → student; EMA → teacher', [('D gradients update B and C','Also learn student tokenizer'),('EMA coefficient .99925','Teacher follows student'),('Resample input and mask','Repeat training step')])])
diagram('representation','Native tokens preserve a spatial or spatio-temporal grid',[
('IMAGE: 256 × 256 pixels, patch 16 × 16', [('2D convolution tokenizer','16 × 16 = 256 tokens'),('Each token has d channels','Shared transformer encoder'),('Modality embedding: image','No repeated static frames')]),
('VIDEO: 16 frames, 256 × 256, tubelet 2', [('3D convolution tokenizer','8 × 16 × 16 = 2048 tokens'),('3D RoPE: time + space','Drop tokens before encoder'),('Mask tokens carry positions','Not observed pixel content')]),
('FOUR FEATURE LEVELS: Nᶜ visible tokens', [('Each level: Nᶜ × d','Concatenate → Nᶜ × 4d'),('MLP fuses channel levels','Predictor gets fused tokens'),('Four targets per position','Match layer to layer')])])
diagram('weighting','Context weights emphasize the edge of a hidden region',[
('TOY 1D GRID: masked site at position 0', [('Visible at 1: distance 1','λ = .5 → weight .5'),('Visible at 4: distance 4','λ = .5 → weight .25'),('Visible at 9: distance 9','λ = .5 → weight .167')]),
('LOSS REDUCTIONS: keep the two branches distinct', [('Masked: average over |M|','Every masked token weight 1'),('Context: average over |C|','Multiply each error by λᵢ'),('Sum branch losses','Not a weighted-mean divide')]),
('PAPER RULE: nearest masked token', [('Distance in token blocks','Spatio-temporal grid'),('λᵢ = λ / sqrt(dmin)','Warm up context coefficient'),('λvideo = .5; λimage = .7','Values from Appendix A')])])
diagram('numeric','One layer, three positions: trace the loss arrows',[
('D TARGETS AND C PREDICTIONS: toy two-channel vectors', [('Mask target (2,1)','Prediction (1.5,1.5)'),('Context 1: (1,0) → (1.2,.2)','Context 2: (0,1) → (.4,.8)'),('L1 errors: 1, .4, .6','Targets → predictions shown')]),
('BRANCH CALCULATIONS: λ = .5, distances 1 and 4', [('Masked mean: 1 / 1 = 1','One masked position'),('Context: (.5×.4+.25×.6)/2','Weighted context mean .175'),('Dense loss = 1.175','Then supervise other levels')]),
('WHY SEPARATE COUNTS MATTER', [('More visible tokens?','Context uses its own count'),('Do not divide by .5+.25','That would give .467'),('Toy L1 sums two channels','No trained model outputs')])])
diagram('timeline','Training carries weights, not labels, between steps',[
('EACH STEP: corruption and a moving latent target', [('Sample image/video + mask','Teacher sees clean input'),('Four dense-level losses','Gradient update student'),('EMA update teacher','Next step repeats')]),
('PRIMARY PHASE: 135k iterations', [('Images 256 × 256','Video 16 frames, 256²'),('12k learning-rate warmup','Then constant learning rate'),('2304 images + 128 clips','Aggregate modality gradients')]),
('COOLDOWN: 12k more iterations', [('Images 512 × 512','Video 64 frames, 384²'),('Learning rate → 1e−6','Higher resolution adaptation'),('Final tables use both phases','Ablations mostly primary only')])])
diagram('paths','The predictor stays or changes depending on the task',[
('DENSE PROBE: frozen representation, trained supervised head', [('Full image → 2D tokenizer','Frozen final-layer encoder'),('Train linear dense head','Depth: learned BatchNorm'),('New image → feature → head','No SSL mask needed')]),
('ACTION ANTICIPATION: retain frozen JEPA predictor', [('Observed clip → encoder','Future positional mask tokens'),('Predict future features','Concat context + prediction'),('Train attentive task probe','Action / verb / noun outputs')]),
('ROBOT MPC: train a different action-conditioned predictor', [('Frozen video features','Droid action data'),('New 24-layer predictor','Simulate candidate actions'),('CEM chooses sequence','Execute, observe, replan')]),
('DISTILLATION: smaller student, fixed ViT-G teacher', [('No deep supervision','Last-layer dense objective'),('New 12-block predictor','Student EMA not in loss'),('EMA student is released','80M or 300M encoder')])])
rows=[('V-JEPA 2',22.2,72.8),('+ Context loss',33.8,62.5),('+ Multi-level prediction',38.6,72.1),('+ VisionMix',40.8,72.6),('+ Native tokenizers',41.4,72.6),('+ Model scaling',47.1,76.1),('+ Cooldown',47.9,77.7)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="740" viewBox="0 0 960 740" role="img" aria-label="Table 1 sequential ablation: ADE20K improves while SSv2 first drops then recovers"><rect width="100%" height="100%" fill="#101c30"/>']
def txt(x,y,s,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
txt(30,42,'Measured · v3 Table 1 · sequential recipe additions',24);txt(30,80,'ADE20K mIoU (%) bars, zero baseline; SSv2 accuracy (%) at right',18)
for i,(name,v,acc) in enumerate(rows):
 y=120+i*72;txt(30,y+25,name,17);p.append(f'<rect x="270" y="{y}" width="{v*8}" height="32" fill="#31859b"/>');txt(670,y+25,str(v));txt(790,y+25,str(acc))
p.append('<path d="M270 650 H670" stroke="#c5d5ee"/>')
for t in range(0,51,10):txt(263+t*8,677,str(t),16)
txt(30,721,'ViT-L start → ViT-G scaling; cooldown is additional training; no error bars.',17)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
assert isclose(.5/sqrt(4),.25)
assert isclose((.5*.4+.25*.6)/2,.175)
assert isclose(1+(.5*.4+.25*.6)/2,1.175)
assert 16//2*(256//16)**2==2048
print('Eight SVGs generated; weighting, loss arithmetic and token-count checks passed.')
