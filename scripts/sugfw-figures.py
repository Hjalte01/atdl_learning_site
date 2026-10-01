"""Teaching diagrams and measured results from supplied arXiv:2608.16110v1."""
from pathlib import Path
from html import escape
from math import exp, log, isclose
out=Path('public/figures/sugfw-plus');out.mkdir(parents=True,exist_ok=True)
def diagram(name,title,rows):
 h=105+len(rows)*145
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{h}" viewBox="0 0 960 {h}" role="img" aria-label="{escape(title)}">','<rect width="100%" height="100%" fill="#101c30"/>','<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="24">{escape(title)}</text>']
 for i,(label,boxes) in enumerate(rows):
  y=85+i*145;p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
  for j,lines in enumerate(boxes):
   x=30+j*310;p.append(f'<rect x="{x}" y="{y+15}" width="275" height="90" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
   for k,line in enumerate(lines):p.append(f'<text x="{x+12}" y="{y+40+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
   if ((name=='training' and i==0) or (name=='architecture' and i==1)) and j==0:
    p.append(f'<path d="M167,{y+106} V{y+120} H787 V{y+106}" fill="none" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
   elif not (name=='training' and i==2) and j<len(boxes)-1:p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
 p.append('</svg>');(out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','One query before any target-domain annotation',[
('CONVENTIONAL MULTI-ROUND AL: an initial target-label set is needed', [('Initial annotated images','Train a task model'),('Score and query pool','Annotate new examples'),('Retrain, then query again','Several annotation rounds')]),
('SUGFW+: pretrained SAM supplies the initial signals', [('Entire target pool unlabeled','SAM is already pretrained'),('Feature + uncertainty','Select one batch of M'),('Human masks → UPFT','Adapt SAM once')]),
('TWO DIFFERENT SAVINGS: no dataset-specific SSL, no test-time TTA', [('Selection still uses K views','Everything-mode masks'),('Training learns a predictor','Imitate prompt embedding'),('Test: one image pathway','No K-view mask generation')])])
diagram('architecture','A → B → C → D: select images, then adapt SAM',[
('A · PFUC: fixed pretrained SAM, no target labels', [('Image X: H × W','Encoder → F: C₀ × H′ × W′'),('K views + automatic masks','Binary unions → mean mask'),('Patch entropy U: H′ × W′','Surrogate, not task truth')]),
('B · PGDR: two summaries of the same patch grid', [('F + U → softmax weights','Weighted sum f in R^C₀'),('U → average pooling','Scalar image uncertainty u'),('One pair (f, u) per image','No annotation yet')]),
('C · GSCU: clusters in f, greedy distances in u', [('C = M feature clusters','One image per cluster'),('Median first; max-min next','Visit random new clusters'),('One query batch D̂','Human target masks y')]),
('D · UPFT: labeled data plus stored uncertainty maps', [('LoRA image encoder','U → uncertainty encoder'),('Prompt decoder; train g','Dice + CE + embedding MSE'),('Adapted automatic SAM','Held-out image → mask')])])
diagram('representation','PFUC close-up: class-agnostic agreement under perturbation',[
('TEACHING PATCH: four views, binary foreground unions', [('View values [1, 1, 0, 0]','Mean foreground q = .5'),('Binary entropy H(.5)','U = .6931 nats'),('Ambiguous under views','No organ label supplied')]),
('STABLE PATCH: agreement can still be consistently wrong', [('View values [1, 1, 1, 1]','Mean foreground q = 1'),('H(1) = 0 by continuity','Lower surrogate uncertainty'),('Not a correctness proof','Large regions filtered out')]),
('MASK TO GRID: the source of the weighting signal', [('Keep each region < HW/2','Union kept regions per view'),('Mean masks → downsample','Match encoder patch grid'),('Compute entropy per patch','Pair U(r) with F(r)')])])
diagram('numeric','Trace a two-patch image through A and B',[
('TOY: scalar features; natural logs; λ = 6', [('Patch 1: q = 1, F = 2','U₁ = 0'),('Patch 2: q = .5, F = 10','U₂ = ln 2 = .6931'),('exp(6U) = [1, 64]','Weights = [1/65, 64/65]')]),
('PGDR: normalized weighted feature versus mean uncertainty', [('f = (2 + 64 × 10) / 65','f = 9.8769'),('u = (0 + .6931) / 2','u = .3466 nats'),('Uniform feature mean = 6','λ emphasizes patch 2')]),
('SOFTMAX LIMITS: feature magnitude is retained', [('Equal U → equal weights','Global average pooling'),('Larger λ → sharper weights','Potential surrogate bias'),('Feature stays a mixture','No new human target')])])
diagram('selection','GSCU: cover the uncertainty spectrum, one per cluster',[
('TOY STEP 1: random cluster A has uncertainties [.1, .3, .5]', [('Start with D⁰ empty','Pick median u = .3'),('Add that image to D¹','Its feature cluster is covered'),('No target mask queried yet','Selection is still offline')]),
('STEP 2: new cluster B has [.2, .6]', [('Distance from .3: [.1, .3]','Max-min chooses .6'),('D² uncertainties [.3, .6]','Two feature clusters covered'),('Not centroid-nearest','Not necessarily highest u')]),
('STEP 3: cluster C has [.05, .4, .65]', [('Nearest distances [.25,.1,.05]','Choose u = .05'),('D³ = [.3, .6, .05]','Uncertainty range expands'),('After M picks: annotate','No retraining between picks')])])
diagram('training','UPFT training: two losses, different targets',[
('SEGMENTATION PATH: the decoder receives the encoded stored map', [('Image → encoder + LoRA','Features F_i'),('Stored U_i → E_un','Prompt embedding Fᵘ_i'),('Decoder(F_i, Fᵘ_i) → P_i','Human y → Dice + CE')]),
('PREDICTOR PATH: learn the prompt available at test time', [('Features F_i + pool mean Ū','Lightweight transformer g'),('Estimated embedding F̂ᵘ_i','Compare with E_un(U_i)'),('Squared embedding error','Ramp-weighted MSE loss')]),
('PARAMETERS: pretrained image weights stay frozen', [('Train rank-32 LoRA','Train mask decoder'),('Train uncertainty encoder','Train uncertainty predictor'),('All share total loss','300 epochs, batch size 16')])])
diagram('inference','Inference: replace the uncertainty-encoder branch',[
('TRAIN-TIME PROMPT: computed from a stored PFUC map', [('K-view U_i is available','Encode with E_un'),('True surrogate embedding','Decoder uses Fᵘ_i'),('g learns to imitate it','Embedding MSE target')]),
('TEST-TIME PROMPT: predicted from the current image', [('New X → adapted encoder','Features F; retain pool Ū'),('g(F, Ū) → F̂ᵘ','Discard uncertainty encoder'),('Decoder(F, F̂ᵘ) → mask','No manual point / box')]),
('SHIFT TO TEST: learned imitation is approximate', [('No fresh K-view U required','No annotation query'),('Ū comes from training pool','Not from test ground truth'),('Prompt error can matter','No calibration guarantee')])])
# Tables 1–4, lowest reported budget. Bars show DSC means only.
rows=[('Promise12 · 3.00%',84.03,86.38),('UTAH · 0.10%',70.96,78.70),('MSD Liver · 0.05%',68.99,82.27),('ISIC 2018 · 0.50%',83.67,86.36)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="610" viewBox="0 0 960 610" role="img" aria-label="Reported DSC means for Random and SUGFW+ at each dataset’s lowest budget"><rect width="100%" height="100%" fill="#101c30"/>']
def txt(x,y,t,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(t)}</text>')
txt(30,40,'Measured · supplied arXiv v1 · Tables 1–4',24);txt(30,75,'DSC (%) · same UPFT model training, different selected images',19)
txt(30,105,'Blue: Random. Green: SUGFW+. Means only; spread is in the guide table.',17)
for i,(name,a,b) in enumerate(rows):
 y=145+i*95;txt(30,y+30,name,18)
 for j,(v,c) in enumerate([(a,'#31859b'),(b,'#56b890')]):
  p.append(f'<rect x="255" y="{y+j*31}" width="{v*5.4}" height="25" fill="{c}"/>');txt(810,y+20+j*31,f'{v:.2f}',18)
p.append('<path d="M255 540 H795" stroke="#c5d5ee"/>')
for tick in range(0,101,20):txt(249+tick*5.4,568,str(tick),16)
txt(30,601,'Budget = fraction of training slices/images, not a fraction of expert hours.',17)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
assert isclose(exp(6*log(2)),64)
assert isclose((2+64*10)/65,9.876923076923077)
assert isclose(log(2)/2,.34657359027997264)
assert [min(abs(u-v) for v in [.3,.6]) for u in [.05,.4,.65]] == [.25,.10000000000000003,.050000000000000044]
assert isclose(.8*.2+.2*.3+.01*.05,.2205)
print('Eight SVGs generated; softmax, entropy, greedy-distance and loss examples verified.')
