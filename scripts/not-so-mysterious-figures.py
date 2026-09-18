"""Reproduce original teaching diagrams; no measured data or model outputs."""
from pathlib import Path
from html import escape
out = Path('public/figures/not-so-mysterious')
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
            if j < len(boxes)-1 and not (name == 'dimension' and i == 0) and not (name == 'architecture' and i == 1 and j == 0): parts.append(f'<path d="M{x+278},{y+49} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    if name == 'architecture': parts.append('<text x="320" y="266" fill="#fff" font-family="sans-serif" font-size="24">+</text>')
    parts.append('</svg>')
    (out/f'{name}.svg').write_text('\n'.join(parts))

diagram('bias','Two ways to express a simplicity bias',[
('Hard restriction', [('Choose degree two','Only terms 1, x, x²'),('Fit available coefficients','Higher terms excluded'),('Cannot adapt beyond it','Even if data need more')]),
('Soft preference', [('Choose degree 150','Many terms available'),('Penalize higher orders','Alternatives cost more'),('Adapt complexity to data','No universal guarantee')])])
diagram('architecture','Fixed features; learned coefficients; two loss terms',[
('A → B → C: prediction', [('A: scalar input x','n examples in a batch'),('B: φ(x) = (1, x, …, xᴶ)','J + 1 fixed coordinates'),('C: weighted sum wᵀφ','One scalar prediction')]),
('Training objective receives two branches', [('Prediction + target y','Squared residual / (2σ²)'),('Coefficients w','Add penalty Σ γʲwⱼ²'),('Total objective L(w)','Differentiate; update w')])])
diagram('paths','Training repeats; prediction holds coefficients fixed',[
('TRAIN: repeat after each update', [('Inputs x + targets y','Fixed polynomial features'),('Predictions → loss','Residual + order penalty'),('Gradient step on w','Carry w to next iteration')]),
('INFER: a new input', [('New scalar x','No target required'),('Same fixed features','Use learned vector w'),('Output wᵀφ(x)','No optimization step')])])
diagram('example','Same endpoint fit; different costs and predictions',[
('Toy training pairs: (−1, −1), (1, 1); degree 3; γ = 2', [('Candidate A: f(x) = x','w = (0, 1, 0, 0)'),('Endpoint residual = 0','Penalty = 2¹ = 2'),('At x = 0.5','Prediction = 0.5')]),
('Compare two candidates, not all possible fits', [('Candidate B: f(x) = x³','w = (0, 0, 0, 1)'),('Endpoint residual = 0','Penalty = 2³ = 8'),('At x = 0.5','Prediction = 0.125')]),
('One gradient step from zero: σ² = 1; learning rate 0.1', [('Residual gradient','(0, −2, 0, −2)'),('New w = (0, .2, 0, .2)','Endpoint predictions ±.4'),('Next penalty gradient','(0, .8, 0, 3.2)')])])
diagram('bound','Certification after fitting: account for the preference',[
('Prior fixed independently of the sample', [('Select fitted hypothesis h','Measure empirical risk'),('Read prior mass P(h)','Cost = log(1 / P(h))'),('Add confidence penalty','Bound expected risk')]),
('Toy: n = 1000; error = .02; δ = .05; loss width = 1', [('Mass 2⁻¹⁰','Penalty ≈ 0.07045'),('Add empirical error .02','Risk bound ≈ 0.09045'),('Mass 2⁻¹⁰⁰ instead','Risk bound ≈ 0.21015')])])
diagram('dimension','Effective dimension: a soft count of spectral directions',[
('Toy PSD spectrum; α = 1; contributions λ / (λ + α)', [('Eigenvalue λ = 9','Contribution = 0.9'),('Eigenvalue λ = 1','Contribution = 0.5'),('Eigenvalue λ = 1/9','Contribution = 0.1')]),
('Interpretation depends on the chosen matrix and scale', [('3 available directions','Effective count = 1.5'),('Change α → change count','Diagnostic, not a bound'),('Hessian is coordinate-based','Flatness is not universal')])])
# Crop the supplied page, preserving its plotted evidence rather than redrawing values.
# Requires Poppler: nix shell nixpkgs#poppler-utils -c python3 scripts/not-so-mysterious-figures.py
import subprocess, math
subprocess.run(['pdftoppm','-f','5','-singlefile','-scale-to','2000','-x','110','-y','100','-W','1350','-H','460','-png','public/papers/not-so-mysterious.pdf',str(out/'source-figure-5')],check=True)
assert abs(.02+math.sqrt((10*math.log(2)+math.log(20))/2000)-.090452)<.00001
assert abs(.02+math.sqrt((100*math.log(2)+math.log(20))/2000)-.21015)<.00001
print('Generated six teaching diagrams and source Figure 5; bound arithmetic checked.')
