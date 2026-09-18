"""Reproduce original teaching diagrams; no measured data or model outputs."""
from pathlib import Path
from html import escape
out = Path('public/figures/selection-under-uncertainty')
out.mkdir(parents=True, exist_ok=True)
def diagram(name, title, rows):
    height = 100 + len(rows)*125
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-label="{escape(title)}">', '<rect width="100%" height="100%" fill="#101c30"/>', '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>', f'<text x="30" y="42" fill="#fff" font-family="sans-serif" font-size="25">{escape(title)}</text>']
    for i,(label,boxes) in enumerate(rows):
        y=85+i*125
        parts.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*310
            parts.append(f'<rect x="{x}" y="{y+15}" width="275" height="68" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines):
                parts.append(f'<text x="{x+14}" y="{y+42+k*24}" fill="#fff" font-family="sans-serif" font-size="18">{escape(line)}</text>')
            if j < len(boxes)-1 and name not in ('map', 'neighbors') and not (name == 'posterior' and i == 2): parts.append(f'<path d="M{x+278},{y+49} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    parts.append('</svg>')
    (out/f'{name}.svg').write_text('\n'.join(parts))


diagram('map','Learning connects design, computation and statistics',[
('Three cooperating roles', [('Design','Which rules are possible?'),('Computation','Which rule can we find?'),('Statistics','How reliable is its score?')]),
('Start with the foundations', [('Chapters 1–2','Learning and validation'),('Sections 3.1–3.3','Concentration tools'),('Sections 4.1–4.4','Selection guarantees')]),
('Then choose a reading branch', [('Sections 4.5–4.11','VC and PAC-Bayes'),('Chapter 7','Online decisions'),('Chapter 5','Regression after Ch. 1–2')])])
diagram('paths','K-NN: construct, select, then freeze for prediction',[
('CONSTRUCT: training data; distance rule fixed externally', [('Store labeled points','S_train = {(xᵢ, yᵢ)}'),('Choose candidate K values','For example: 1, 3, 5'),('Candidate rules h_K','Same stored examples')]),
('SELECT: use separate validation examples and their labels', [('Compute predictions','For every candidate K'),('Compare validation loss','Choose a rule'),('Freeze K and stored data','Evaluate on untouched test')]),
('INFER: new input x; no new target label', [('Compute distances','Use stored xᵢ and rule d'),('Sort and take K nearest','Retrieve their labels yᵢ'),('Sum labels; take sign','Return prediction h_K(x)')])])
diagram('neighbors','Representation close-up: the stored points are the model',[
('Toy query x = 0.2; distance = absolute difference', [('Point x₁ = 0','Label +1; distance 0.2'),('Point x₂ = 1','Label −1; distance 0.8'),('Point x₃ = 2','Label −1; distance 1.8')]),
('Already sorted by distance; compare two votes', [('K = 1','Sum = +1 → predict +1'),('K = 3','Sum = −1 → predict −1'),('Future correctness?','Needs a new target label')])])
diagram('posterior','Soft selection: carry a distribution into prediction',[
('LEARN: prior π fixed independently of the certification sample', [('Observe sample losses','Evaluate candidate rules'),('Choose posterior ρ','Fit versus KL(ρ || π)'),('Carry forward ρ','Not necessarily Bayes')]),
('RANDOMIZED INFERENCE: Definition 4.25', [('Sample h from ρ','A fresh draw per prediction'),('Observe new input x','Apply sampled rule h'),('Output h(x)','Bound average risk over ρ')]),
('TOY: two equal-error classifiers; π = (1/2, 1/2)', [('Keep ρ = (1/2, 1/2)','KL = 0'),('Concentrate ρ = (1, 0)','KL = ln 2'),('Majority vote differs','Needs Section 4.9 analysis')])])
diagram('online','The feedback protocol determines the next update',[
('FULL INFORMATION: all losses revealed after choosing', [('Draw action A_t','From probabilities p_t'),('Observe every loss','ℓ_t,1 through ℓ_t,K'),('Update every total','Carry C_t forward')]),
('BANDIT: only chosen action loss revealed', [('Draw action A_t','From probabilities p_t'),('Observe only ℓ_t,A_t','Other losses hidden'),('Estimate loss vector','Importance weighting')]),
('REPEAT: online action and learning remain interleaved', [('Past totals C_(t−1)','Compute next p_t'),('Choose → observe','One round of feedback'),('New totals C_t','Next round t + 1')])])
diagram('trace','Hedge over three rounds: η = ln 2; two actions',[
('ROUND 1: begin with cumulative losses C₀ = (0, 0)', [('p₁ = (1/2, 1/2)','Observe losses (0, 1)'),('Expected loss = 1/2','New totals C₁ = (0, 1)'),('Weights (1, 1/2)','Normalize → p₂ = (2/3, 1/3)')]),
('ROUND 2: lower previous loss gets more probability', [('p₂ = (2/3, 1/3)','Observe losses (1, 0)'),('Expected loss = 2/3','New totals C₂ = (1, 1)'),('Weights (1/2, 1/2)','Normalize → p₃ = (1/2, 1/2)')]),
('ROUND 3: compare accumulated loss to one fixed action', [('p₃ = (1/2, 1/2)','Observe losses (0, 1)'),('Expected total = 5/3','Best fixed total = 1'),('Expected regret = 2/3','Sampled runs may differ')])])
# A theoretical plot, calculated from Eq. (4.4), not empirical performance.
import math
n, delta = 1000, .05
margins=[math.sqrt(math.log(m/delta)/(2*n)) for m in [1,100,1000000]]
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="Calculated finite-class Hoeffding margins"><rect width="960" height="420" fill="#101c30"/>']
def text(x,y,s,size=20):
    parts.append(f'<text x="{x}" y="{y}" fill="white" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
text(30,42,'Selection cost grows with candidate count',25)
text(30,78,'Calculated margins: n = 1000; δ = 0.05; losses in [0, 1]',20)
for i,(m,v) in enumerate(zip([1,100,1000000],margins)):
    y=115+75*i
    text(30,y+26,f'M = {m:,}')
    parts.append(f'<rect x="220" y="{y}" width="{v*6000:.3f}" height="40" rx="5" fill="{["#31859b","#8162b0","#379c77"][i]}"/>')
    text(235+v*6000,y+26,f'{v:.4f}')
parts.append('<path d="M220 333 H820" stroke="#c5d5ee"/>')
for tick in [0,.02,.04,.06,.08,.1]:
    x=220+tick*6000
    parts.append(f'<path d="M{x} 333 v7" stroke="#c5d5ee"/>')
    text(x-12,362,f'{tick:.2f}',16)
text(220,400,'Additive upper-bound margin (not measured error)',18)
parts.append('</svg>')
(out/'bounds.svg').write_text('\n'.join(parts))
assert abs(.05+margins[1]-.1116478)<1e-6
assert abs(.1+math.sqrt(math.log(2*math.sqrt(n)/delta)/(2*n))-.159765)<1e-5
print('Generated seven SVGs; certificate arithmetic checked.')
