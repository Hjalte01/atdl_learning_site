"""Reproduce Flow Matching schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/flow-matching')
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

diagram('paths','Choose distributions first; learn how to move between them',[
('CONDITIONAL · fix one endpoint z; vary the initial noise', [('t = 0: ε ~ N(0,I)','Every endpoint starts here'),('0 < t < 1: tz + (1−t)ε','Gaussian cloud around tz'),('t = 1: z','Cloud collapses to this point')],True),
('MARGINAL · average over data endpoints z', [('Initial Gaussian p₀','Same starting distribution'),('Mixture pₜ over all z','Not generally a Gaussian'),('Data distribution p₁','Many possible outputs')],True)])
diagram('training','Training: build a known velocity target without an ODE',[
('A → B → C · fixed path recipe; trainable velocity network', [('A: data z, noise ε, time t','z, ε ∈ ℝᵈ; t is a scalar'),('B: xₜ = tz + (1−t)ε','Network inputs: xₜ and t'),('C: uθ(xₜ,t) ∈ ℝᵈ','One predicted velocity / entry')],True),
('TARGET → LOSS → UPDATE · endpoint z does not enter network', [('v = z − ε ∈ ℝᵈ','Known from sampled pair'),('L = ||uθ(xₜ,t) − v||²','Compare C with this target'),('Gradient update of θ','Resample z, ε, t next batch')],True)])
diagram('posterior','One location: several possible endpoints, one mean velocity',[
('TOY · t = 0.5, x = 0.5; equal priors on z = −1 and z = +1', [('Endpoint −1','u(x|−1) = −3'),('Endpoint +1','u(x|+1) = +1'),('Posterior weights','0.1192 and 0.8808')],False),
('MARGINAL FIELD · posterior weights depend on current x and t', [('Weighted negative target','0.1192 × (−3) = −0.3576'),('Weighted positive target','0.8808 × 1 = 0.8808'),('Sum: u(x,t) ≈ 0.5232','Not uniform mean −1')],False)])
diagram('numbers','Trace one training example through A → B → C → loss',[
('A · a two-dimensional teaching example', [('z = (2, −1)','ε = (−2, 1)'),('t = 0.25','xₜ = (−1, 0.5)'),('v = z − ε = (4, −2)','This is velocity, not position')],True),
('C · stipulate a prediction to see the loss', [('uθ = (3, −1)','Prediction error = (−1, 1)'),('Squared-error sum = 2','Mean over 2 entries = 1'),('Update network weights','No sampled ODE needed')],True)])
diagram('sampling','Inference: freeze θ; carry the generated state forward',[
('GENERAL · the network does not know a target endpoint z', [('Draw X₀ ~ N(0,I)','Set t = 0'),('Evaluate uθ(Xₜ,t)','Xnext = Xₜ + h uθ(Xₜ,t)'),('Carry Xnext and t+h','Repeat to t = 1')],True),
('ORACLE TOY · constant velocity (4,−2), step h = 0.25', [('X₀ = (−2,1)','X₀.₂₅ = (−1,0.5)'),('X₀.₅ = (0,0)','X₀.₇₅ = (1,−0.5)'),('X₁ = (2,−1)','Exact only for this toy field')],True)])
diagram('sde','Same time-marginals need not mean the same trajectories',[
('ODE · learned velocity field; random initialization', [('X₀ drawn once','Freeze trained θ'),('Drift u(Xₜ,t)','No new Brownian noise'),('Desired marginal pₜ','Exact-field idealization')],True),
('SDE · pair the added noise with a score correction', [('Same initial distribution','Choose noise schedule σₜ'),('Drift u + (σₜ²/2)s','Noise increment σₜ√h ξ'),('Same desired marginal pₜ','Different random trajectories')],True),
('NUMERIC · u = 0.5; score s = −2; σ = 1; h = 0.04; ξ = 0.3', [('Corrected drift = −0.5','Drift increment = −0.02'),('Noise increment = 0.06','√0.04 × 0.3'),('Net increment = 0.04','An illustrative single draw')],False)])
diagram('latent','Latent generation: compress, learn a flow, then decode',[
('AUTOENCODER TRAINING · a separate stage', [('Image I','Encoder produces latent z'),('Reconstruct image D(z)','Balance fidelity / regularity'),('Trained encoder + decoder','Freeze for this staged recipe')],True),
('FLOW TRAINING · use encoded data as endpoints', [('Latent z + noise ε + t','Mix into latent state xₜ'),('Time / prompt aware model','Predict latent velocity'),('Regress against z − ε','Update flow network θ')],True),
('INFERENCE · no source image required for text-to-image', [('Latent noise + prompt y','Prompt fixed during sampling'),('ODE → generated latent','State evolves, θ stays fixed'),('Decoder → output image','Compression limits fidelity')],True)])
assert tuple(.25*z+.75*e for z,e in zip((2,-1),(-2,1)))==(-1,.5)
p=1/(1+math.exp(2)); assert math.isclose(-3*p+1-p,.5231883119115297)
assert math.isclose((.5+.5*(-2))*.04+math.sqrt(.04)*.3,.04)
