"""Reproducible CEC teaching schematics; historical results from course Table 1."""
from pathlib import Path
from html import escape
from math import log, exp
out = Path('public/figures/cec'); out.mkdir(parents=True, exist_ok=True)
def diagram(name, title, rows):
    height = 105 + len(rows)*145
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-label="{escape(title)}">', '<rect width="100%" height="100%" fill="#101c30"/>', '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="24">{escape(title)}</text>']
    for i,(label,boxes) in enumerate(rows):
        y=85+i*145
        p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*310
            p.append(f'<rect x="{x}" y="{y+15}" width="275" height="90" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines):
                p.append(f'<text x="{x+12}" y="{y+40+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
            if name == 'architecture' and i == 0 and j == 0:
                p.append(f'<path d="M{x+137},{y+106} V{y+120} H787 V{y+106}" fill="none" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
            elif j<len(boxes)-1:
                p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    p.append('</svg>'); (out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','Which images deserve the next few labels?',[
('RANDOM: covers the pool without consulting the model', [('Unlabeled pool D_U','Known class names'),('Sample b images','No uncertainty ranking'),('Oracle reveals labels','Tune prompts on D_L')]),
('RAW ENTROPY: may reflect model bias or isolated confusion', [('CLIP probabilities P','Unequal class tendencies'),('Rank by H(x)','May query similar outliers'),('Oracle reveals labels','No diversity guarantee')]),
('CEC: correct scores, use neighbors, cover feature regions', [('Calibrated entropy H-hat','Pool-derived Q(i)'),('Neighbor term → U(x)','Weighted b-way clustering'),('Nearest to each center','Query b labels; tune')])])
diagram('architecture','CEC architecture: two encoders, one acquisition path',[
('A · REPRESENT: image and text meet in a shared feature space', [('Image x → frozen f','z in R^d; unit norm'),('Context V + class C_i','Frozen h → text t_i in R^d'),('Cosine / T → softmax','P(x) in R^K; T = .01')]),
('B · SCORE: labels are still hidden', [('Per-class top N in D_U','Q(i) → P-hat → H-hat'),('k nearest image features','Distance-weighted H_NN'),('U = H-hat + H_NN','One weight per image')]),
('C · QUERY: geometry plus uncertainty', [('Features + weights U','Weighted K-means'),('b centers → b images','Nearest real image per center'),('Oracle returns true y','D_L grows; D_U shrinks')]),
('D · LEARN: labeled route feeds A on the next cycle', [('All collected (x,y)','Raw CLIP probabilities P'),('Cross-entropy loss','Gradient through frozen h'),('Update context V only','Recompute acquisition')])])
diagram('paths','Selection, prompt training and inference have separate jobs',[
('SELECTION: no true targets from the remaining pool', [('Current CLIP + prompts','Images and class names'),('Calibration / neighbors','Weighted clusters'),('Query oracle labels','Add to accumulated D_L')]),
('TRAINING: reload zero-shot CLIP each AL round (§5)', [('Collected D_L','Initialize CoOp context'),('Frozen f and h','Differentiate loss wrt V'),('Learn context vectors V','Not encoder weights')]),
('INFERENCE: classify a new image; no oracle or clusters', [('New image → f(x)','Learned V + class → h'),('Cosine / T → softmax','K class probabilities'),('Argmax → class','Evaluate held-out accuracy')])])
diagram('calibration','Calibration changes relative evidence, not oracle truth',[
('Q(i): estimate separately for each class from top N predictions', [('Class 1: top scores','Mean Q(1) = .8'),('Class 2: top scores','Mean Q(2) = .4'),('No labels used','Q need not sum to 1')]),
('TOY QUERY IMAGE: normalize after dividing by Q', [('Raw P = (.8, .2)','H = .5004 nats'),('P / Q = (1, .5)','Sum = 1.5'),('P-hat = (2/3, 1/3)','H-hat = .6365 nats')]),
('SAME Q, DIFFERENT IMAGE: entropy need not increase', [('Raw P = (.5, .5)','H = .6931 nats'),('P / Q = (.625, 1.25)','Class 2 gains relative weight'),('P-hat = (1/3, 2/3)','H-hat = .6365 nats')])])
diagram('neighbors','Neighborhood uncertainty: an attenuated average',[
('TOY: k = 2, alpha = 1; distances are squared unit-feature distances', [('Neighbor 1','distance² = .1','H-hat = .6'),('Neighbor 2','distance² = .4','H-hat = .4'),('Weighted contributions','.5429 and .2681','Divide sum by k = 2')]),
('ADD: no product, threshold or hard outlier veto', [('H_NN = .4055','Not divided by weight sum'),('Self H-hat = .6365','U = 1.0420'),('Uncertain neighborhood','Can raise acquisition weight')]),
('REPRESENTATION: geometry is distinct from class probabilities', [('z = f(x) / norm(f(x))','Unit image feature in R^d'),('P-hat in R^K','Estimated class distribution'),('Oracle y','True target revealed later')])])
diagram('clustering','Weighted centers move before representative selection',[
('TEACHING PROJECTION: one cluster, two positions (not CLIP vectors)', [('Position a = 0','Weight U_a = 1'),('Position b = 2','Weight U_b = 3'),('Weighted mean = 1.5','(1×0 + 3×2) / 4')]),
('QUERY THE NEAREST REAL POINT, NOT THE CENTER ITSELF', [('Distance from a = 1.5','Distance from b = .5'),('Query image b','Then reveal its label'),('Full CEC: b clusters','One representative each')]),
('DIFFERENT RULE: cluster first, then maximize uncertainty', [('Unweighted center = 1','Different partition geometry'),('Highest U in each cluster','Greedy alternative'),('Not CEC selection rule','Compare in source Figure 3')])])
diagram('loop','AL state over time: toy 1,000-image initial pool',[
('CYCLE 1: choose b = 10; initial D_L is empty', [('D_L = 0; D_U = 1000','Zero-shot probabilities'),('CEC selects 10 IDs','Oracle returns 10 targets'),('D_L = 10; D_U = 990','Reset / train CoOp')]),
('CYCLE 2: new prompts change probabilities and scores', [('Recompute P and Q','Image encoder stays frozen'),('CEC selects 10 new IDs','Oracle returns targets'),('D_L = 20; D_U = 980','Reset / train on all 20')]),
('CYCLE 6: fixed toy batch size, accumulated supervision', [('60 total queried labels','D_U = 940'),('Train on all 60','No guaranteed class balance'),('Held-out inference','No query on test set')])])
rows=[('1%',38.4,.2,47.9,1.2),('2%',44.2,2.9,52.8,1.0),('5%',54.1,2.9,58.2,2.0)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="520" viewBox="0 0 960 520" role="img" aria-label="Textures held-out accuracy: Random and CEC with three-seed standard deviations"><rect width="100%" height="100%" fill="#101c30"/>']
def t(x,y,s,size=19):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
t(30,40,'Measured · Table 1 · Textures · CoOp / CLIP ViT-B/16',24)
t(30,74,'Held-out accuracy (%); mean ± SD over three seeds; zero-based bars',18)
for i,(budget,r,rs,c,cs) in enumerate(rows):
    y=110+105*i; t(30,y+22,f'Budget {budget}')
    for offset,value,sd,color in [(0,r,rs,'#31859b'),(35,c,cs,'#9274ce')]:
        yy=y+offset
        p.append(f'<rect x="190" y="{yy}" width="{value*9}" height="26" fill="{color}"/>')
        xx=190+value*9
        p.append(f'<path d="M{xx-sd*9},{yy+13} H{xx+sd*9} M{xx-sd*9},{yy+6} V{yy+20} M{xx+sd*9},{yy+6} V{yy+20}" stroke="white" stroke-width="2"/>')
        t(780,yy+20,f'{value:.1f} ± {sd:.1f}',18)
for tick in range(0,61,10):t(180+tick*9,450,str(tick),16)
p.append('<path d="M190 423 H730" stroke="#c5d5ee"/>');t(30,490,'Teal: Random     Purple: CEC     Zero-shot reference: 44.3%',19)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
h=-(2/3*log(2/3)+1/3*log(1/3));nn=(exp(-.1)*.6+exp(-.4)*.4)/2
assert round(h,4)==.6365 and round(nn,4)==.4055 and round(h+nn,4)==1.0420
print('Generated eight CEC figures; teaching arithmetic verified.')
