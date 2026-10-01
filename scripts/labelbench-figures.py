"""Original LabelBench teaching schematics and selected Table 2 measurements."""
from pathlib import Path
from html import escape
out = Path('public/figures/labelbench'); out.mkdir(parents=True, exist_ok=True)
def diagram(name, title, rows):
    height=100+len(rows)*140
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-label="{escape(title)}">', '<rect width="100%" height="100%" fill="#101c30"/>', '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>', f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="25">{escape(title)}</text>']
    for i,(label,boxes) in enumerate(rows):
        y=85+i*140
        p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*310
            p.append(f'<rect x="{x}" y="{y+15}" width="275" height="85" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines): p.append(f'<text x="{x+12}" y="{y+39+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
            if j<len(boxes)-1 and name!='representations': p.append(f'<path d="M{x+278},{y+57} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    p.append('</svg>'); (out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','Three decisions share one annotation budget',[
('TRANSFER: where does the representation come from?', [('Pretrained encoder','CLIP or CoCa ViT-B32'),('Downstream images','Use existing visual features'),('Useful starting point','Still needs adaptation')]),
('SEMI-SUPERVISION: how do unqueried inputs contribute?', [('Labeled examples L','Supervised loss'),('Unlabeled examples U','Pseudo-label / consistency'),('Combined training','Human labels stay distinct')]),
('ACTIVE LEARNING: which labels should we buy next?', [('Current model','Predict on U'),('Acquisition rule','Rank or diversify examples'),('Human/oracle annotation','Grow L; reduce U')])])
diagram('architecture','Selection architecture: expensive features, cheap updates',[
('PRECOMPUTE: encoder frozen; five seeded augmentation sets', [('Image a_s(x)','Pixels → encoder E_phi'),('Cached z_s(x) in R^d','Store feature vectors'),('Reuse over epochs','No encoder backprop')]),
('TRAIN PROXY: reset head; labels + unlabeled consistency', [('Cached features z','Sample augmentation sets'),('Head h_theta','R^d → R^K logits'),('Softmax p in R^K','Loss updates theta only')]),
('SELECT: target labels remain hidden until this last box', [('Features and p on U','Acquisition score / geometry'),('Batch Q of size b','Send selected IDs to oracle'),('Reveal true y for Q','Carry new L and U forward')])])
diagram('paths','Three paths: selection training, final training, prediction',[
('REPEATED SELECTION: phi fixed; proxy theta learned afresh', [('Cached L and U','Train small proxy'),('Select and annotate Q','Update dataset partitions'),('Next annotation round','New proxy; larger L')]),
('FINAL TRAINING: discard selection-only restriction', [('Collected L + remaining U','Pretrained CLIP / CoCa'),('End-to-end FlexMatch','Update encoder and head'),('Final predictor f','Used for reported evaluation')]),
('INFERENCE: no acquisition or oracle', [('New test image x','Encoder E_phi-final'),('Classifier head','Softmax / argmax'),('Predicted class','Compare to test ground truth')])])
diagram('representations','Features, predictions and labels have different roles',[
('TOY: same three-class probabilities; different feature locations', [('Example A','z = (0, 0)','p = (.45, .40, .15)'),('Example B','z = (0.1, 0)','p = (.45, .40, .15)'),('Example C','z = (5, 5)','p = (.45, .40, .15)')]),
('WHAT EACH ARRAY CAN SAY', [('Feature z in R^d','Diversity / distances','Not a class label'),('Probability p in R^K','Margin / entropy','Not guaranteed calibrated'),('Oracle label y','Revealed after selection','Not a cached prediction')])])
diagram('trace','One toy round: choose one label, then retrain',[
('PROXY SCORES: b = 1; smallest top-two margin wins', [('A: (.80, .15, .05)','Margin = .65'),('B: (.45, .40, .15)','Margin = .05'),('C: (.60, .20, .20)','Margin = .40')]),
('ANNOTATE B: model said class 1; oracle may disagree', [('Select B','Highest entropy too'),('Oracle returns class 2','Human label overrides guess'),('L grows by 1','U shrinks by 1')]),
('NEXT ROUND: changed training data, changed scores', [('Reset and train proxy','Use new L and remaining U'),('Recompute predictions','Do not reuse stale scores'),('Continue to budget','Then train final full model')])])
diagram('loop','Annotation state over time: toy budget B = 6, batch b = 2',[
('START: random initial batch counts toward budget', [('Pool of 10 inputs','No labels visible'),('Query 2 at random','L = 2; U = 8'),('Train proxy from scratch','Score remaining U')]),
('ROUND 2: adaptive acquisition', [('Query 2 selected IDs','Reveal oracle labels'),('L = 4; U = 6','Training can still use U'),('Reset / retrain proxy','Choose next batch')]),
('ROUND 3: budget exhausted', [('Query 2 selected IDs','Total human labels = 6'),('L = 6; U = 4','Stop annotation'),('Final full fine-tuning','Use both partitions')])])
# Selected historical Table 2 measurements, not a reproduction of the benchmark.
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540" role="img" aria-label="Table 2 test and pool accuracy for Random and Confidence selection"><rect width="960" height="540" fill="#101c30"/>']
def t(x,y,s,size=20): p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
t(30,40,'ImageNet · CLIP ViT-B32 · 400,000 labels · Table 2',24)
t(30,75,'Final end-to-end FlexMatch; two trials; bars start at zero',18)
rows=[('Random',76.12,83.35),('Confidence / full',77.38,90.11),('Confidence / shallow',76.96,88.93),('Confidence / linear',77.23,89.01)]
for i,(label,test,pool) in enumerate(rows):
    y=110+85*i; t(30,y+22,label,18)
    for offset,val,color in [(0,test,'#31859b'),(30,pool,'#9274ce')]:
        p.append(f'<rect x="270" y="{y+offset}" width="{val*5.5}" height="24" fill="{color}"/>'); t(280+val*5.5,y+offset+19,f'{val:.2f}%',17)
for tick in range(0,101,20): t(260+tick*5.5,473,str(tick),16)
p.append('<path d="M270 445 H820" stroke="#c5d5ee"/>'); t(30,515,'Teal: held-out test accuracy    Purple: whole-pool accuracy',19)
p.append('</svg>'); (out/'results.svg').write_text('\n'.join(p)+'\n')
assert abs((.45-.40)-.05)<1e-10
print('Generated seven LabelBench SVGs.')
