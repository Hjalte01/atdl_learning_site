"""Active Prompt teaching schematics and supplied TMLR 10/2025 Table 2 results."""
from pathlib import Path
from html import escape
from math import log, isclose
out=Path('public/figures/active-prompt'); out.mkdir(parents=True,exist_ok=True)
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
            if name=='architecture' and i==0:
                if j==0:p.append(f'<path d="M167,{y+106} V{y+120} H787 V{y+106}" fill="none" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
                elif j==1:p.append(f'<path d="M618,{y+60} H645" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
            elif j<len(boxes)-1:p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    p.append('</svg>');(out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','One candidate need not cost one human annotation',[
('RANDOM + ALL HUMAN: no task-guided selection', [('Unlabeled image pool','Known target class names'),('B random candidates','Query all B labels'),('Train a small prompt','Human bill = B')]),
('CB + SQ: task-guided coverage, then an annotation gate', [('Image + weighted text','Class-guided clusters'),('B representative images','Check class thresholds'),('Human or pseudo target','Human bill ≤ B')]),
('RISK: saved annotations are not verified labels', [('Confident prediction','Can still be incorrect'),('Pseudo target enters D_r','May affect next threshold'),('Track quality and cost','No correctness certificate')])])
diagram('architecture','A → B → C → D: select data, then learn prompts',[
('A · PARALLEL ENCODERS: neither encoder trains', [('Image x → frozen image θ','I(x) in R^d'),('Context + class → frozen θ','T_c in R^d, c in C'),('Cosine / τ → softmax','p(x) in R^|C|')]),
('B · REPRESENT: use p as mixture weights, retain the image', [('Sum p_c × T_c','Weighted text in R^d'),('Concatenate [I, weighted T]','F_C(x) in R^(2d)'),('K-means on all images','Representatives → Q_r')]),
('C · GATE: use the preceding round’s model and data', [('D_(r−1) + prompt t_(r−1)','Mean confidence per class'),('Predicted class + confidence','Compare with ε_(r,c)'),('Below: oracle; else: pseudo','Accumulate targets in D_r')]),
('D · LEARN: new prompt becomes next round’s selection model', [('Randomly reset prompts','Retain encoder parameters'),('Cross-entropy on all D_r','Gradient → context vectors'),('New learned prompt t_r','Return to A next round')])])
diagram('representation','Close-up: soft class evidence becomes a geometric feature',[
('TOY: d = 2, two classes; values are illustrative encoder outputs', [('Image I = (1, 0)','Class A text T_A = (1, 0)'),('Class B text T_B = (0, 1)','Probabilities p = (.8, .2)'),('Weighted T = (.8, .2)','F_C = (1, 0, .8, .2)')]),
('SAME IMAGE BLOCK, DIFFERENT SOFT EVIDENCE', [('Another feature has I=(1,0)','Text mixture = (.2, .8)'),('ΔF = (0, 0, .6, −.6)','Squared distance = .72'),('Text evidence separates them','No true labels used')]),
('KEEP BOTH BLOCKS: texture and class priors remain distinct', [('I: visual representation','May encode background'),('Weighted T: class mixture','May inherit CLIP bias'),('Concatenate, not sum','No hard one-hot label')])])
diagram('selection','Cluster-balanced means coverage of feature clusters',[
('ROUND 1: K = B; one centroid-nearest image per cluster', [('Cluster members F_C(x)','Mean → centroid c_i'),('Minimize distance to c_i','Choose a real image x_i*'),('B clusters → B candidates','All B receive oracle labels')]),
('TOY CLUSTER: one-dimensional projection only', [('Positions 0, 2, 3','Centroid = 5/3'),('Distances 5/3, 1/3, 4/3','Choose position 2'),('Representative ≠ centroid','Neither is an oracle label')]),
('LATER: K = r × B; clusters change with new prompts', [('Recluster image set','Avoid covered clusters'),('Intended Q_r size = B','Favor large uncovered groups'),('Clusters ≠ true classes','Top-K wording needs care')])])
diagram('thresholds','Same confidence, different predicted class, different bill',[
('PREVIOUS TRAINING DATA: average assigned-label probability', [('Class A confidences .8, .9','ε_A = .85'),('Class B confidences .5, .7','ε_B = .60'),('Includes old pseudo targets','Threshold is not calibration')]),
('CANDIDATE GATE: confidence = probability of predicted class', [('Predict A with q = .80','q < .85 → oracle'),('Predict B with q = .80','q ≥ .60 → pseudo B'),('Predict A with q = .85','Equality → pseudo A')]),
('ROUND 1 AND EMPTY CLASSES', [('No previous D_0','Query every first-round ID'),('If a class has no examples','Eq. 11 denominator is zero'),('No fallback specified in PDF','Do not invent one')])])
diagram('paths','Training and inference use different information',[
('SELECTION: preceding learned prompt is fixed while choosing', [('Old prompt + unlabeled pool','Feature mixtures / clusters'),('Q_r + class thresholds','Oracle or pseudo target'),('D_r grows by B examples','Keep old assigned targets')]),
('TRAINING: supervised CE with mixed target provenance', [('Reset t_r; frozen encoders','All accumulated D_r'),('Loss −log p(target | x)','No gate confidence weighting'),('Update prompt inputs only','200 epochs in main setup')]),
('INFERENCE: classify a new held-out image', [('New image → image encoder','Final prompt + class names'),('Cosine scores → softmax','Argmax class prediction'),('No clusters or human query','No gradient step')])])
diagram('loop','Toy trace: B = 2, growing data and a smaller human bill',[
('ROUND 1: no class thresholds yet', [('D_0 empty; K = 2','Use hand-crafted prompt'),('2 selected IDs → 2 queries','Assume one A and one B'),('|D_1| = 2; humans = 2','Reset / train t_1')]),
('ROUND 2: use ε_A = .85 and ε_B = .60 from the worked example', [('K = 4; select 2 new IDs','A .80 and B .80'),('A → human; B → pseudo','One new paid annotation'),('|D_2| = 4; humans = 3','Reset / train t_2 on all 4')]),
('BUDGET AND LOSS: pseudo targets also influence learning', [('4 targets, 3 human labels','Toy saving = 25%'),('If training p(target)=.8','CE = −ln(.8) = .2231'),('Wrong pseudo still gets CE','Confidence is not truth')])])
# Source Table 2; zero-based bars, reported mean ± standard deviation.
rows=[('Image only',52.72,.37,'#31859b'),('Weighted text only',48.24,.21,'#9274ce'),('Image + soft text',55.92,.38,'#56b890'),('Oracle label-balanced',48.10,.41,'#ad8d58'),('Image + true-label text',58.27,.43,'#ad8d58')]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" viewBox="0 0 960 600" role="img" aria-label="Table 2 initial-round average accuracy across seven datasets"><rect width="100%" height="100%" fill="#101c30"/>']
def text(x,y,s,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
text(30,40,'Measured · TMLR 10/2025 · Table 2 · initial acquisition',24)
text(30,75,'Average accuracy (%) across seven datasets; CLIP ViT-B/32',19)
text(30,104,'Mean ± SD, three trials. Last two rows use ground-truth labels.',18)
for i,(name,value,sd,color) in enumerate(rows):
    y=140+72*i;text(30,y+23,name,17);p.append(f'<rect x="270" y="{y}" width="{value*7}" height="30" fill="{color}"/>');x=270+value*7
    p.append(f'<path d="M{x-sd*7},{y+15} H{x+sd*7} M{x-sd*7},{y+8} V{y+22} M{x+sd*7},{y+8} V{y+22}" stroke="white" stroke-width="2"/>');text(780,y+23,f'{value:.2f} ± {sd:.2f}')
p.append('<path d="M270 510 H760" stroke="#c5d5ee"/>')
for tick in range(0,71,10):text(264+tick*7,535,str(tick),16)
text(30,575,'This tests feature choice, not selective-querying savings or equal paid budgets.',18)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
assert isclose(.8+.2,1) and isclose((.8-.2)**2+(.2-.8)**2,.72)
assert isclose((.8+.9)/2,.85) and isclose((.5+.7)/2,.6)
assert isclose(-log(.8),.2231435513142097) and isclose(1-3/4,.25)
print('Wrote eight figures; feature, threshold, cost and CE arithmetic verified.')
