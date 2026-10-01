"""Reproducible CSCS teaching schematics and Table 2 chart; arXiv v1."""
from pathlib import Path
from html import escape
from math import sqrt, isclose
out=Path('public/figures/cscs');out.mkdir(parents=True,exist_ok=True)
def diagram(name,title,rows):
 h=110+len(rows)*145
 p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{h}" viewBox="0 0 960 {h}" role="img" aria-label="{escape(title)}">','<rect width="100%" height="100%" fill="#101c30"/>','<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="white" font-family="sans-serif" font-size="24">{escape(title)}</text>']
 for i,(label,boxes) in enumerate(rows):
  y=85+i*145;p.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
  for j,lines in enumerate(boxes):
   x=30+j*310;p.append(f'<rect x="{x}" y="{y+15}" width="275" height="90" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
   for k,line in enumerate(lines):p.append(f'<text x="{x+12}" y="{y+40+k*23}" fill="white" font-family="sans-serif" font-size="17">{escape(line)}</text>')
   if j<len(boxes)-1 and not (name=='architecture' and i==1):p.append(f'<path d="M{x+278},{y+60} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
 p.append('</svg>');(out/f'{name}.svg').write_text('\n'.join(p)+'\n')
diagram('problem','The first batch is a separate decision',[
('RANDOM INITIALIZATION: no informed acquisition', [('Unlabeled 3D volumes','Sample B uniformly'),('Expert masks for B scans','Train task segmenter'),('Predict held-out masks','Variable first-set quality')]),
('WARM-START AL: labels and a task model already exist', [('Initial labeled set L₀','Train task segmenter'),('Score unlabeled pool','Query another batch'),('Retrain and repeat','An iterative process')]),
('CSCS: learn signals without task labels, then query once', [('Unlabeled training pool','Self-supervised learning'),('Pool-aware clustered query','One annotation batch'),('Train first nnU-Net','Later AL not evaluated')])])
diagram('architecture','CSCS: global adaptation, local selection',[
('A · SSL MODEL: train on the unlabeled pool, then freeze', [('Xᵢ: 3D image tensor','Masked multi-view inputs'),('Swin-UNETR SSL','Encoder + reconstruction'),('Embedding zᵢ in R⁷⁶⁸','R reconstruction views')]),
('B · TWO BRANCHES: derive scalars from the frozen outputs', [('Reconstruction views','Variance → raw Uᵢ'),('Embeddings + k neighbors','Inverse distance → raw Tᵢ'),('Pool min-max scaling','Normalized Uᵢ and Tᵢ')]),
('C · QUERY: combine geometry, budget and cluster-local scores', [('Ranks → DCR; B / √N','Pacing → global γ'),('B clusters; merge small','Sᵢ = Tᵢ^(1−γ) Uᵢ^γ'),('Winners + global fill','B scan identifiers')]),
('D · TASK MODEL: human labels arrive only here', [('Expert volume masks','Selected images + labels'),('Separate 3D nnU-Net','Supervised training'),('Held-out volume → mask','No query score at test')])])
diagram('representation','Two scalars describe different properties',[
('RECONSTRUCTION DISAGREEMENT: schematic single-voxel views', [('Values [.3, .5]','Mean .4'),('Population variance .01','Average across voxels'),('U: difficulty proxy','Not segmentation entropy')]),
('LOCAL DENSITY: schematic neighbor distances', [('Distances [.2, .3]','Mean distance .25'),('Inverse mean = 4','Raw typicality T'),('Close to many neighbors','Not an anatomical label')]),
('NORMALIZATION: compute across the entire unlabeled pool', [('Subtract pool minimum','Divide by pool range'),('Separate scaling for U, T','Each in [0, 1]'),('Constant range is undefined','V1 omits safeguards')])])
diagram('pacing','DCR sets direction; budget sets strength',[
('POSITIVE ALIGNMENT: uncertainty and typicality ranks agree', [('DCR > 0','Difficult scans are central'),('γ > .5','Favor uncertainty more'),('B increases → γ increases','Within conservative range')]),
('NEGATIVE ALIGNMENT: uncertainty rises in peripheral regions', [('DCR < 0','Difficult scans are isolated'),('γ < .5','Favor typicality more'),('B increases → γ decreases','Not always harder-first')]),
('NEUTRAL AND LIMITS: finite positive budgets', [('DCR = 0 → γ = .5','Any annotation budget'),('m = (B/√N)/(1+B/√N)','m stays between 0 and 1'),('.25 < γ < .75','Not an optimality theorem')])])
diagram('selection','After merging, one-per-cluster is only the first pass',[
('ILLUSTRATIVE B = 3: six volumes before merging', [('Cluster A: 3 scans','One primary winner'),('Cluster B: 2 scans','Too small: merge'),('Cluster C: 1 scan','Too small: merge')]),
('NEAREST-CENTROID MERGING: illustration ends in one cluster', [('B and C join A','M = 1 surviving cluster'),('Choose its max-score scan','One query slot occupied'),('B − M = 2 slots remain','Same γ and same scores')]),
('GLOBAL FILL: no retraining between the individual picks', [('Rank all unselected scans','Take next two scores'),('Exactly B = 3 identifiers','May share a region'),('Experts annotate the batch','Only now supervised labels')])])
gamma=.5+.25*(1/sqrt(3))/(1+1/sqrt(3));score=(2/3)**(1-gamma)*.25**gamma
diagram('numeric','A coherent toy pool → one selected scan',[
('A → B: unit embeddings at angles 0°, 60°, 180°; toy k = 2', [('Pairwise distances','1, 2, √3'),('Raw T ≈ [.667,.732,.536]','Normalized T = [⅔,1,0]'),('Raw U = [.01,.04,0]','Normalized U = [.25,1,0]')]),
('C: N = 3, B = 1; ranks agree perfectly', [('DCR = 1','Effective budget = .5774'),('Confidence m = .3660',f'γ = {gamma:.4f}'),(f'Scores ≈ [{score:.4f},1,0]','Single cluster → B wins')]),
('D: query the scan named B, not a predicted mask', [('Expert supplies mask','No task labels used earlier'),('Train downstream nnU-Net','Illustrative pool only'),('Population variance assumed','Not a model result')])])
diagram('paths','Weight updates and test prediction happen in separate paths',[
('SELF-SUPERVISED TRAINING: no annotation masks', [('Masked training images','Reconstruction task'),('Update SSL model weights','Swin-UNETR framework'),('Freeze for acquisition','Store embeddings / scores')]),
('SUPERVISED TRAINING: selected volumes only', [('Fixed query identifiers','Expert image-mask pairs'),('Update nnU-Net weights','3D full resolution'),('Trained task segmenter','250-epoch protocol')]),
('INFERENCE: acquisition statistics are no longer needed', [('Held-out 3D volume','Standard preprocessing'),('Trained nnU-Net forward','Predict voxel labels'),('Evaluate against reference','Dice and boundary metrics')])])
rows=[('BraTS · 116 / 387',85.2,85.4),('FeTA · 12 / 39',64.1,73.7),('Spleen · 10 / 34',93.3,95.7),('DIANE · 7 / 18',75.3,83.4)]
p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="610" viewBox="0 0 960 610" role="img" aria-label="Table 2 Dice means at middle nominal budget: Random versus CSCS"><rect width="100%" height="100%" fill="#101c30"/>']
def txt(x,y,t,size=18):p.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(t)}</text>')
txt(30,40,'Measured · supplied arXiv v1 · Table 2',24);txt(30,75,'Dice (%) · middle nominal operating point · means over 3 runs',19)
txt(30,105,'Blue: Random. Green: CSCS. No error bars inferred from rounded means.',17)
for i,(name,a,b) in enumerate(rows):
 y=145+i*95;txt(30,y+30,name,18)
 for j,(v,c) in enumerate([(a,'#31859b'),(b,'#56b890')]):
  p.append(f'<rect x="255" y="{y+j*31}" width="{v*5.4}" height="25" fill="{c}"/>');txt(810,y+20+j*31,f'{v:.1f}',18)
p.append('<path d="M255 540 H795" stroke="#c5d5ee"/>')
for tick in range(0,101,20):txt(249+tick*5.4,568,str(tick),16)
txt(30,601,'Counts shown explicitly: DIANE 7/18 = 38.89%, although table labels this 30%.',17)
p.append('</svg>');(out/'results.svg').write_text('\n'.join(p)+'\n')
assert isclose(gamma,.5915063509461096)
assert isclose((2/(1+sqrt(3))-2/(2+sqrt(3)))/(2/(1+sqrt(3))-2/(2+sqrt(3))),1)
assert isclose((2/3-2/(2+sqrt(3)))/(2/(1+sqrt(3))-2/(2+sqrt(3))),2/3)
print(f'Toy gamma={gamma:.8f}, S_A={score:.8f}; eight SVGs generated.')
