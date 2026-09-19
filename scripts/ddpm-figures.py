"""Reproduce DDPM schematics and check teaching arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/ddpms')
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

diagram('architecture','One shared noise predictor, two uses of its output',[
('TRAINING · image tensors retain H × W × C coordinates', [('A: clean image x₀','Pixels scaled to [−1,1]'),('B: fixed corruption','Add known Gaussian noise'),('C: noise predictor εθ','Input: xₜ and time t')],True),
('C · FUNCTIONAL U-NET · weights shared across all times', [('Down / up feature paths','Group normalization'),('Sinusoidal time embedding','Attention at 16 × 16'),('Image-shaped prediction','Same shape as xₜ')],True),
('OUTPUT BRANCHES · loss versus state update', [('C: predicted noise','One output tensor'),('D: training squared error','Update network weights'),('E: reverse mean + noise','Update state; freeze weights')],False)])
diagram('representation','Image-shaped states, changing information',[
('TEACHING NOISE LEVELS · cumulative retained signal ᾱ', [('ᾱ = 1','Signal 1; noise 0'),('ᾱ = 0.64','Signal 0.8; noise 0.6'),('ᾱ = 0.01','Signal 0.1; noise ≈ 0.995')],True),
('SAME DIMENSIONS · no autoencoder compression', [('x₀: clean pixels','H × W × C'),('xₜ: noisy pixels','H × W × C'),('xT: near Gaussian','H × W × C')],True),
('DO NOT CONFUSE · variance factors and amplitudes', [('βₜ: one-step noise variance','αₜ = 1 − βₜ'),('ᾱₜ: product of α values','Accumulates corruption'),('Square roots: amplitudes','Not pixels replaced')],False)])
diagram('paths','Training takes a shortcut; generation takes the chain',[
('TRAIN · one random time per example', [('A: sample x₀, t, ε','t uniform from 1 to T'),('B → C: construct xₜ','Predict known target ε'),('D: squared noise error','Gradient update to θ')],True),
('GENERATE · trained θ stays fixed', [('Draw xT from N(0,I)','No original image input'),('C → E: reverse transition','Fresh noise for t > 1'),('Carry xₜ₋₁ forward','Repeat at time t − 1')],True),
('COMPUTE · original paper setup', [('Training corruption','Direct marginal formula'),('Generation from scratch','1,000 network evaluations'),('Final step t = 1','Display mean; z = 0')],False)])
diagram('sampling','The reverse loop changes state, not network weights',[
('ONE ITERATION · from time t to time t − 1', [('Current state xₜ and t','C: predict εθ(xₜ,t)'),('E: convert noise to μθ','Use αₜ, βₜ and ᾱₜ'),('Sample xₜ₋₁ = μθ + σₜz','z fresh; zero at t = 1')],True),
('REPEAT · output becomes the next input', [('xT → xT₋₁','Start near pure noise'),('… → x₂ → x₁','Reuse the same network'),('x₁ → displayed x₀','No final random increment')],True),
('OPTIONAL PREVIEW · distinct from the next state', [('C: predicted noise','Combine with current xₜ'),('Estimate clean x̂₀','Eq. 15; may amplify errors'),('Not a full sampler','Do not replace xₜ₋₁ with x̂₀')],True)])
diagram('worked','One coordinate through B, C, D and E',[
('CONSTRUCTED EXAMPLE · ᾱₜ = 0.64; noise draw ε = 0.2', [('A: x₀ = 0.8','B: 0.8 × 0.8 + 0.6 × 0.2'),('Noisy xₜ = 0.76','C predicts ε̂ = 0.1'),('D: (0.2 − 0.1)² = 0.01','Clean preview x̂₀ = 0.875')],True),
('INTERMEDIATE STEP · βₜ = 0.1; αₜ = 0.9; σₜ² = 0.1', [('E: subtract (0.1/0.6) × 0.1','Then divide by √0.9'),('Reverse mean ≈ 0.7835','Fresh draw z = −0.32'),('Add √0.1 × (−0.32)','Next state ≈ 0.6823')],True),
('THREE DIFFERENT QUANTITIES', [('Clean estimate: 0.875','Prediction of final image'),('Reverse mean: 0.7835','Center of one transition'),('Next state: 0.6823','One random transition draw')],False)])
diagram('progression','A hierarchy of information, not a pixel filling order',[
('REVERSE DIRECTION · schematic reading of Figures 5–7', [('Early: low signal','Broad structure emerges'),('Middle: more information','Shape becomes clearer'),('Late: fine detail','Small residual uncertainty')],True),
('TWO INTERPRETATIONS · keep their inputs distinct', [('Generation','Start from random xT'),('Progressive decoding','Receive information about x₀'),('Both can preview x̂₀','Using the current xₜ')],False),
('REPORTED CIFAR10 ACCOUNTING · §4.3', [('Rate: 1.78 bits/dim','Intermediate bound terms'),('Distortion: 1.97 bits/dim','Final decoder term'),('Total: 3.75 bits/dim','Not an actual file size')],True)])
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="400" viewBox="0 0 1000 400" role="img" aria-label="Table 2 unconditional CIFAR10 FID ablation"><rect width="100%" height="100%" fill="#101c30"/><g fill="white" font-family="sans-serif"><text x="30" y="42" font-size="25">CIFAR10: parameterization and loss must be read together</text><text x="30" y="82" font-size="19">Table 2 · unconditional · training-set reference · FID lower is better</text>']
for i,(label,val) in enumerate([('Mean prediction + bound',13.22),('Noise prediction + bound',13.51),('Noise prediction + simple loss',3.17)]):
 y=120+i*75
 parts.append(f'<text x="30" y="{y+23}" font-size="18">{label}</text><rect x="340" y="{y}" width="{val/15*550}" height="35" fill="{["#719bdb","#ac96df","#63d9b2"][i]}"/><text x="910" y="{y+23}" font-size="20">{val}</text>')
parts.append('<text x="340" y="365" font-size="17">Linear bar scale: 0–15. Fixed reverse covariance.</text></g></svg>')
(out/'results.svg').write_text('\n'.join(parts))
xt=.8*.8+.6*.2
mean=(xt-.1/.6*.1)/math.sqrt(.9)
state=mean+math.sqrt(.1)*(-.32)
assert math.isclose(xt,.76)
assert math.isclose((xt-.6*.1)/.8,.875)
assert round(mean,4)==.7835 and round(state,4)==.6823
print('Seven DDPM figures generated; scalar forward/reverse arithmetic checked.')
