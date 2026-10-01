"""Reproduce nnActive teaching schematics and selected TMLR 08/2025 Table 3 data."""
from pathlib import Path
from html import escape
from math import log, exp, isclose
out=Path('public/figures/nnactive');out.mkdir(parents=True,exist_ok=True)
def diagram(name,title,rows):
    height=105+len(rows)*145
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-label="{escape(title)}">','<rect width="100%" height="100%" fill="#101c30"/>','<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="24">{escape(title)}</text>']
    for i,(label,boxes) in enumerate(rows):
        y=85+i*145
        p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*310
            p.append(f'<rect x="{x}" y="{y+15}" width="275" height="90" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines):p.append(f'<text x="{x+12}" y="{y+40+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
            if j<len(boxes)-1:p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    p.append('</svg>');(out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','Same patch count, different information and human effort',[
('PLAIN RANDOM: a large background class dominates the volume', [('Random location in scan','Often empty background'),('Cheap annotation','Little foreground learned'),('Low Dice at equal patches','A weak baseline for AL')]),
('FOREGROUND-AWARE RANDOM: simulate rough structure screening', [('Some fully random patches','Some foreground / borders'),('Diverse foreground classes','More delineation work'),('Stronger Dice baseline','Still count effort separately')]),
('UNCERTAINTY: model predictions guide annotation', [('Score ambiguous regions','May repeat similar borders'),('Annotate selected patches','More compute per round'),('Test across tasks / budgets','No universal superiority')])])
diagram('architecture','A → B → C → D: probabilities become new partial labels',[
('A · TRAINING: input patch geometry comes from nnU-Net plans', [('Volume X: M × H × W × D','Query labels + valid mask'),('3D nnU-Net fθ: context','5 cross-validation models'),('p_m(c | v): K classes','Partial loss → model weights')]),
('B · ACQUISITION: run predictions on training / pool images', [('Five probability volumes','Average p̄: K × H × W × D'),('Voxel entropy or BALD','Mean over query patch'),('Overlap filter within image','Rank across images')]),
('C · ANNOTATION: only selected spatial regions acquire labels', [('Choose n eligible patches','Main study overlap o = 0'),('Reveal segmentation masks','Simulated expert annotation'),('Accumulate label locations','Old masks persist')]),
('D · NEXT ROUND: replace fitted weights, retain labeled regions', [('Reprocess partial targets','Sample label + context'),('Retrain from scratch','Main study: 200 epochs'),('New predictions → B','Final model → held-out test')])])
diagram('representation','Missing target is distinct from known background',[
('TOY FOUR-VOXEL VIEW: a projection of a 3D training patch', [('Image visible at all voxels','[x₁, x₂, x₃, x₄]'),('Targets [FG, BG, ?, ?]','Valid mask [1, 1, 0, 0]'),('Loss only at v₁ and v₂','Context can use all x')]),
('TEACHING CE COMPONENT: not the full nnU-Net loss recipe', [('p(FG | v₁) = .8','−ln(.8) = .2231'),('p(BG | v₂) = .9','−ln(.9) = .1054'),('Mean labeled CE = .1643','No target loss at v₃, v₄')]),
('REGION SAMPLING: retain labels without forcing crop centers', [('At least one labeled voxel','Inside the training crop'),('Random valid position','More surrounding context'),('Query size ≠ forward size','Both are 3D patches')])])
diagram('uncertainty','Same predictive entropy, different ensemble disagreement',[
('TOY TWO-MODEL BINARY VOXEL: nats, natural logarithms', [('Model 1: (.9, .1)','Model 2: (.1, .9)'),('Mean p̄ = (.5, .5)','PE = H(p̄) = .6931'),('Mean H(p_m) = .3251','BALD = .3681')]),
('AGREEMENT ON AMBIGUITY: uncertain does not imply disagreement', [('Both models: (.5, .5)','Same mean p̄ = (.5, .5)'),('PE remains .6931','Mean H = .6931'),('BALD becomes zero','Different acquisition score')]),
('PATCH CLOSE-UP: average voxel scores before ranking', [('Voxel PE [.6931, .3251]','Toy patch has two voxels'),('Patch mean = .5091','Not sum or maximum'),('Main study uses 5 models','Noise can diversify selection')])])
diagram('selection','A greedy patch trace: overlap first, global rank second',[
('TOY IMAGE A: scores follow the uncertainty box above', [('Patch A1 score .5091','Patch A2 score .4900'),('A2 overlaps A1','o = 0 → reject A2'),('A3 score .3000','Disjoint → retain A3')]),
('TOY IMAGE B: spatial overlap is defined within each image', [('Patch B1 score .4500','Disjoint from old labels'),('Combine A1, A3, B1','Global descending scores'),('n = 2 → select A1, B1','Annotate these regions')]),
('WHY A FILTER IS NOT FULL DIVERSITY', [('Avoid exact spatial overlap','Retain old query locations'),('Disjoint can look alike','Even across different scans'),('Noise adds diversity','No class-coverage guarantee')])])
diagram('paths','Training, selection and test inference have different roles',[
('TRAIN: fit every model using only available segmentation targets', [('Image crop + partial mask','Unlabeled image context'),('Partial supervised loss','Gradient updates full model'),('Weights replaced each round','No frozen CLIP backbone')]),
('SELECT: fixed ensemble predicts pool regions', [('p₁ … p₅ volumes','Compute U(v), aggregate'),('Overlap + rank / perturb','Acquire new segmentation'),('Append labels to state','No gradient during ranking')]),
('TEST: fixed fitted models segment new complete scans', [('Held-out scan X','nnU-Net patchwise inference'),('Voxel class prediction','Compare to dense test mask'),('Mean Dice evaluates quality','No annotation or retraining')])])
diagram('loop','ACDC Low: five evaluated budgets, 150 queried patches',[
('SHARED START: labels are selected before the first model', [('30 patches = 20% of 150','All classes in ≥ 2 patches'),('Remainder: Random 33% FG','Same initialization protocol'),('Train / evaluate at 30','Four data-related seeds')]),
('ITERATE: acquire 30 more patches between evaluated budgets', [('Predict → score → select','Reveal 30 new patch masks'),('Retain old labels / locations','Retrain from scratch'),('Evaluate at 60, 90, 120','Then final model at 150')]),
('DESIGN PARAMETERS: query size and patch size differ', [('Query size = 30 patches','Geometry = 4 × 40 × 40'),('Smaller batch: more rounds','Halving axes: 1/8 voxels'),('Effort ≠ patch count alone','Compute and FG also matter')])])
# All five methods in Table 3, KiTS Medium. ΔFinal Dice ×100, not absolute Dice.
rows=[('BALD',1.73),('PowerBALD',2.30),('SoftrankBALD',1.41),('Predictive Entropy',.58),('PowerPE',1.89)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="570" viewBox="0 0 960 570" role="img" aria-label="KiTS Medium Table 3 gains from 500 epoch acquisition versus precomputed queries"><rect width="100%" height="100%" fill="#101c30"/>']
def text(x,y,s,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
text(30,40,'Measured · Table 3 · KiTS Medium · 500-epoch comparison',23)
text(30,77,'ΔFinal Dice × 100: 500 Epochs minus Precomputed query trajectory',18)
text(30,106,'Both evaluation models train 500 epochs; acquisition models differ.',18)
for i,(name,value) in enumerate(rows):
    y=140+65*i;text(30,y+23,name,18);p.append(f'<rect x="260" y="{y}" width="{value*220}" height="30" fill="#56b890"/>');text(800,y+23,f'+{value:.2f}')
p.append('<path d="M260 475 H810" stroke="#c5d5ee"/>')
for tick in [0,.5,1,1.5,2,2.5]:text(253+tick*220,501,str(tick),16)
text(30,543,'Reported gains; this chart omits source significance shading (α = .1).',18)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
# FG-Eff toy fit curve, with explicitly synthetic endpoints.
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="470" viewBox="0 0 960 470" role="img" aria-label="Synthetic foreground efficiency curves with gamma 5 and 10"><rect width="100%" height="100%" fill="#101c30"/>']
text(30,40,'Toy FG-Eff · same endpoints, faster gap decay at larger γ',24)
text(30,76,'y_full = .8; y(t₀) = .5; t₀ = .1. No measured dataset values.',18)
for gamma,color in [(5,'#9274ce'),(10,'#56b890')]:
    pts=[]
    for i in range(101):
        t=.1+.3*i/100;y=.8-.3*exp(-gamma*(t-.1));pts.append(f'{120+2200*(t-.1):.2f},{350-700*(y-.5):.2f}')
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="4"/>')
    text(800,150 if gamma==10 else 205,f'γ = {gamma}',20)
p.append('<path d="M120 120 V350 H780" fill="none" stroke="#c5d5ee"/>')
for t in [.1,.2,.3,.4]:text(110+2200*(t-.1),380,str(t),17)
for y in [.5,.6,.7,.8]:text(65,355-700*(y-.5),str(y),17)
text(350,412,'Foreground fraction t',18);text(30,110,'Dice',18)
text(30,449,'At t = .2: y = .6180 (γ=5), .6896 (γ=10). Same model / budget only.',18)
p.append('</svg>');(out/'efficiency.svg').write_text('\n'.join(p)+'\n')
H=lambda ps:-sum(p*log(p) for p in ps if p)
assert isclose(H([.5,.5]),.69314718056,abs_tol=1e-10)
assert isclose(H([.5,.5])-H([.9,.1]),.36806420717,abs_tol=1e-10)
assert isclose((-log(.8)-log(.9))/2,.164252033486)
assert isclose((H([.5,.5])+H([.9,.1]))/2,.5091150769757)
print('Wrote nine SVGs; entropy, disagreement, patch aggregation and masked CE arithmetic verified.')
