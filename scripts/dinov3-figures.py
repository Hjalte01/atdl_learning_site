"""DINOv3 v1 teaching diagrams; rerun from repository root."""
from pathlib import Path
from html import escape
from math import isclose
out=Path('public/figures/dinov3');out.mkdir(parents=True,exist_ok=True)
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
diagram('problem','One image, two kinds of representation',[
('GLOBAL TASK: recognize the scene', [('Image of a bird on a branch','CLS summarizes the image'),('Global classifier','Can improve with training'),('Is a bird present?','No location required')]),
('DENSE TASK: preserve spatial distinctions', [('Patch grid of same image','Local feature at each patch'),('Similarities or dense head','Needs coherent local features'),('Which patches are the bird?','Can degrade during training')])])
diagram('architecture','A · crops → B · backbone → C · three learning signals',[
('STUDENT: gradient updates; 2 global + 8 local crops per image', [('Global: 256 × 256 pixels','Local: 112 × 112 pixels'),('ViT-7B / 16; d = 4096','CLS + 4 registers + patches'),('CLS head → DINO','Masked patch head → iBOT')]),
('EMA TEACHER: no gradient; clean global crops only', [('Same image, global views','No student patch masking'),('EMA copy of student','Teacher feature outputs'),('Balanced head targets','Match views / masked sites')]),
('UPDATE: loss arrows converge here', [('DINO + iBOT targets','DKoleo on student CLS'),('LPre = LD + Li + .1 LK','Backpropagate to student'),('θT ← .999 θT + .001 θS','Teacher follows student')])])
diagram('representation','Patch tokens are vectors; Gram entries are relationships',[
('256-PIXEL GLOBAL CROP: patch size 16', [('16 × 16 patch grid','P = 256 patch positions'),('Feature matrix X: 256 × 4096','Normalize rows to unit length'),('G = X Xᵀ: 256 × 256','Not a 4096 × 4096 matrix')]),
('ONE ENTRY: row i, column j', [('Patch i vector xᵢ','Patch j vector xⱼ'),('Gᵢⱼ = xᵢ · xⱼ','Cosine similarity in [−1,1]'),('Diagonal entries = 1','No semantic labels given')]),
('EXCLUDED FROM THIS MATRIX', [('CLS: image representation','Registers: 4 extra tokens'),('Gram uses patch features','Global crops only'),('Match all patch pairs','Not only spatial neighbors')])])
diagram('gram','Refinement adds a second, stable teacher branch',[
('STUDENT BRANCH: same global crop content', [('Standard-resolution crop','Student patch features XS'),('Row normalization','GS = XS XSᵀ'),('Differentiable Gram loss','Updates student weights')]),
('GRAM TARGET BRANCH: no gradient', [('Early EMA snapshot','Initial dense-feature anchor'),('2× image → Gram teacher','Bicubic feature downsample'),('Normalize → XG → GG','Fixed between refreshes')]),
('COMPARE: retain existing objectives', [('GS and GG: P × P','Squared Frobenius difference'),('LRef = wD LD + Li','+ wDK LK + wGram LGram'),('EMA teacher still active','Distinct from Gram teacher')])])
diagram('numeric','Two patches trace the Gram branch exactly',[
('NORMALIZED PATCH FEATURES: toy P = d = 2', [('XG rows: (1,0), (0,1)','Target patches orthogonal'),('XS rows: (1,0), (.6,.8)','Student rows length 1'),('GG = [[1,0],[0,1]]','GS = [[1,.6],[.6,1]]')]),
('COMPARE THE MATRICES', [('GS − GG = [[0,.6],[.6,0]]','Diagonal contributes zero'),('Squared sum = .36 + .36','LGram = .72'),('wGram = 2 → 1.44','Contribution to total loss')]),
('FREEDOM: shared orthogonal rotation', [('Rotate both student rows','XS becomes XS Q'),('Q Qᵀ = I','XS Q Qᵀ XSᵀ = GS'),('Same Gram, new coordinates','No feature equality required')])])
diagram('timeline','Three clocks: student, EMA teacher and Gram teacher',[
('INITIAL PRETRAINING: 1M iterations, web model', [('Student: gradient update','DINO + iBOT + DKoleo'),('EMA: update every step','Early snapshot saved'),('No Gram objective yet','Dense quality may decline')]),
('REFINEMENT: add Gram anchoring', [('Start from late student','Initialize early Gram teacher'),('Gradient + EMA continue','Gram fixed per interval'),('Every 10k: copy EMA to Gram','At most 3 updates; App. C')]),
('HIGH-RESOLUTION ADAPTATION: additional 10k steps', [('Student global 512 or 768','Local crop sizes also vary'),('7B teacher as Gram target','App. C lists crop mixtures'),('Improved high-res features','Not an infinite guarantee')])])
diagram('paths','Training choices differ from downstream inference',[
('DISTILLATION: learned smaller student, fixed 7B teacher', [('Frozen 7B teacher forward','Shared across student groups'),('DINO + iBOT + DKoleo','No Gram term here'),('Small student weight updates','1M + 250k cooldown')]),
('DENSE LINEAR PROBE: annotations used for head training', [('Image → frozen backbone','Patch feature grid'),('Train linear prediction head','Backbone weights fixed'),('Image → features → head','Segmentation or depth')]),
('TEXT ALIGNMENT: separate caption-supervised stage', [('Frozen vision backbone','Two added transformer layers'),('CLS + mean patch concat','Learn text representation'),('Contrastive image-text match','Not base SSL pretraining')])])
rows=[('DINOv2 g/14',49.5),('Web-DINO 7B/14',42.7),('AM-RADIOv2.5 g/14',53.0),('DINOv3 7B/16',55.9)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="580" viewBox="0 0 960 580" role="img" aria-label="Table 3 ADE20k linear probing mIoU: four frozen backbones"><rect width="100%" height="100%" fill="#101c30"/>']
def txt(x,y,t,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(t)}</text>')
txt(30,42,'Measured · DINOv3 v1 Table 3 · ADE20k',24);txt(30,80,'Frozen backbone + trained linear head · mIoU (%) · higher is better',19)
for i,(name,v) in enumerate(rows):
 y=120+i*90;txt(30,y+28,name);p.append(f'<rect x="275" y="{y}" width="{v*8}" height="38" fill="{"#56b890" if i==3 else "#31859b"}"/>');txt(760,y+28,str(v))
p.append('<path d="M275 490 H835" stroke="#c5d5ee"/>')
for t in range(0,71,10):txt(270+t*8,520,str(t),16)
txt(30,562,'1024 patch tokens per model; selected baselines; no error bars supplied.',17)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
assert isclose(.6**2+.8**2,1)
assert isclose(2*.6**2,.72)
assert isclose(2*.72,1.44)
print('Eight diagrams generated; toy normalization and Gram loss assertions passed.')
