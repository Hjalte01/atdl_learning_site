"""Reproduce GAN schematics and check teaching arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/gans')
out.mkdir(parents=True, exist_ok=True)
def diagram(name, title, rows):
    height = 110 + len(rows)*125
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-label="{escape(title)}">', '<rect width="100%" height="100%" fill="#101c30"/>', '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>', f'<text x="30" y="42" fill="#fff" font-family="sans-serif" font-size="25">{escape(title)}</text>']
    for i,(label,boxes,arrows) in enumerate(rows):
        y=85+i*125
        parts.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*325
            parts.append(f'<rect x="{x}" y="{y+15}" width="290" height="68" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines):
                parts.append(f'<text x="{x+12}" y="{y+42+k*24}" fill="#fff" font-family="sans-serif" font-size="17">{escape(line)}</text>')
            if arrows and j < len(boxes)-1: parts.append(f'<path d="M{x+294},{y+49} H{x+319}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</svg>')
    (out/f'{name}.svg').write_text('\n'.join(parts))


diagram('architecture','Learn a sampling map with a learned training signal',[
('FAKE PATH · z ∈ ℝᵈ → generated x ∈ ℝᴰ → one scalar', [('A: noise prior pz','Sample z'),('B: generator Gθ','Produce Gθ(z)'),('C: discriminator Dφ','Probability of real origin')],True),
('REAL PATH · training data enters the same discriminator C', [('Real minibatch x','x ∈ ℝᴰ'),('C: same Dφ(x)','Real target = 1'),('D: classification objective','Fake target = 0')],True),
('GRADIENT PATH · alternating players, shared differentiable connection', [('Update φ: hold θ fixed','Real and fake scores'),('Update θ: hold φ fixed','Backprop C → B'),('Generate: keep only A → B','No discriminator required')],False)])
diagram('paths','Training alternates weights; sampling just transforms noise',[
('DISCRIMINATOR STEP · ascend log D(x) + log(1 − D(G(z)))', [('Draw real x and noise z','Compute fake G(z)'),('Evaluate C on both','Different origin labels'),('Update φ only','Generator weights fixed')],True),
('GENERATOR STEP · practical loss −log D(G(z))', [('Draw fresh noise z','Compute fake G(z)'),('Differentiate through C','Do not update φ'),('Update θ only','Keep gradient to G(z)')],True),
('SAMPLING · after training', [('Draw new noise z','No real image input'),('One forward pass Gθ(z)','Weights fixed'),('Return synthetic image','No iterative denoising')],True)])
diagram('representation','Noise coordinates are not image labels',[
('IMPLICIT DISTRIBUTION · probability mass moves through G', [('Many draws z ∼ pz','Simple noise distribution'),('Same learned mapping G','Different input coordinates'),('Samples define pg','Density need not be explicit')],True),
('TEACHING EXAMPLE · z uniform on [0,1], scalar G(z) = z²', [('z = 0, ¼, ½, ¾, 1','Evenly spaced inputs'),('x = 0, ¹⁄₁₆, ¼, ⁹⁄₁₆, 1','Unevenly spaced outputs'),('P(x ≤ ¼) = ½','Half the mass near zero')],True),
('COLLAPSE · a different map G(z) = c', [('All input coordinates','Even distant z values'),('Same output c','No output diversity'),('May fool a weak D','Does not match diverse data')],True)])
diagram('worked','One scalar generator step, traced through B and C',[
('CONSTRUCTED NETWORKS · Gθ(z) = θz; D(x) = sigmoid(2x − 2)', [('A: z = 1; θ = 0','B: fake x = 0'),('C: logit a = −2','D(fake) ≈ 0.1192'),('Loss = −log D(fake)','≈ 2.1269')],True),
('BACKWARD · discriminator slope is 2; z is 1', [('∂L/∂a = D − 1','≈ −0.8808'),('∂L/∂θ = (D − 1) × 2 × 1','≈ −1.7616'),('Step size η = 0.1','θ new ≈ 0.1762')],True),
('RECOMPUTE WITH SAME z · pedagogical check, not a full training run', [('B: fake x ≈ 0.1762','Generator changed'),('C: D(fake) ≈ 0.1614','Discriminator unchanged'),('Loss ≈ 1.8237','One local improvement')],True)])
diagram('gradients','Why the practical generator objective changes',[
('LET a BE THE FAKE LOGIT · D = sigmoid(a)', [('D = 0.01','Strong rejection'),('D = 0.5','Uncertain origin'),('D = 0.99','Strong acceptance')],False),
('MINIMAX · derivative of log(1 − D) with respect to a', [('−D = −0.01','Weak early signal'),('−D = −0.5','Intermediate signal'),('−D = −0.99','Strong signal')],False),
('NON-SATURATING · derivative of −log D with respect to a', [('D − 1 = −0.99','Strong early signal'),('D − 1 = −0.5','Intermediate signal'),('D − 1 = −0.01','Weak signal near success')],False)])
diagram('equilibrium','An ideal discriminator compares local probability mass',[
('TWO-BIN TEACHING DISTRIBUTIONS · not discrete GAN training', [('Data: (0.75, 0.25)','Generator: (0.25, 0.75)'),('D* = data / (data + gen)','D* = (0.75, 0.25)'),('C(G) ≈ −1.1247','Above optimum −1.3863')],True),
('MATCHED DISTRIBUTIONS · generator now equals data', [('Data: (0.75, 0.25)','Generator: (0.75, 0.25)'),('D* = (0.5, 0.5)','Origins indistinguishable'),('C(G) = −log 4','Jensen–Shannon = 0')],True),
('THEOREM CONDITIONS · distribution-space result', [('Unlimited capacity','Optimal discriminator'),('Small distribution updates','Not arbitrary SGD steps'),('Finite networks differ','No practical convergence proof')],False)])
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="570" viewBox="0 0 1000 570" role="img" aria-label="Table 1 Parzen log-likelihood estimates for MNIST and TFD"><rect width="100%" height="100%" fill="#101c30"/><g fill="white" font-family="sans-serif"><text x="30" y="40" font-size="23">Reported evidence: fitted Parzen density, not exact GAN likelihood</text>']
for panel,(dataset,vals,errors,maximum) in enumerate([('MNIST (real-valued)',[138,121,214,225],[2,1.6,1.1,2],250),('Toronto Face Database',[1909,2110,1890,2057],[66,50,29,26],2300)]):
 y0=90+panel*230
 parts.append(f'<text x="30" y="{y0}" font-size="21">{dataset} · separate scale 0–{maximum}</text>')
 for i,(name,val,err) in enumerate(zip(['DBN','Stacked CAE','Deep GSN','Adversarial nets'],vals,errors)):
  y=y0+20+i*42; w=val/maximum*560; e=err/maximum*560
  parts.extend([f'<text x="30" y="{y+20}" font-size="18">{name}</text>',f'<rect x="210" y="{y}" width="{w}" height="27" fill="{"#63d9b2" if i==3 else "#719bdb"}"/>',f'<path d="M{210+w-e},{y+13} H{210+w+e}" stroke="white" stroke-width="3"/>',f'<text x="800" y="{y+20}" font-size="18">{val} ± {err}</text>'])
parts.append('<text x="30" y="555" font-size="17">Table 1 · standard errors: examples for MNIST, folds for TFD · higher is better</text></g></svg>')
(out/'results.svg').write_text('\n'.join(parts))
sig=lambda a:1/(1+math.exp(-a))
theta=.1*2*(1-sig(-2))
assert math.isclose(theta,.1761594155955765)
assert .161 < sig(2*theta-2) < .162
assert math.isclose(1.5*math.log(.75)+.5*math.log(.25),-1.1246702892376166)
print('Seven GAN figures generated; teaching arithmetic checked.')
