"""Reproduce Deep Generative Modeling schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/deep-generative-modeling')
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


diagram('map','Six ways to make a distribution computationally usable',[
('EXPLICIT DENSITIES · choose a tractable computation', [('Autoregressive','Multiply conditionals'),('Normalizing flow','Invert + volume correction'),('VAE','Bound a latent integral')],False),
('OTHER OBJECTIVES · change what the network learns', [('Energy model','Unnormalized preference'),('GAN','Learn through a critic'),('Score model','Learn a density gradient')],False)])
diagram('autoregressive','Same network, different information at training and generation',[
('TRAIN · observed x = (1,0,1); causal mask prevents future access', [('Prefix: empty','P(x₁=1) = 0.6'),('Prefix: (1)','P(x₂=0 | 1) = 0.75'),('Prefix: (1,0)','P(x₃=1 | 1,0) = 0.8')],True),
('GENERATE · freeze θ; sampled prefix becomes the next input', [('Sample first symbol','Carry sampled x₁'),('Sample next conditional','Carry sampled (x₁,x₂)'),('Repeat to sequence end','Toy sequence mass = 0.36')],True)])
diagram('flow','A flow preserves probability mass, not density height',[
('GENERATE · a one-dimensional affine teaching example', [('z ~ Normal(0,1)','Base density at z=0: 0.399'),('x = 2z + 1','Intervals double in width'),('x = 1 at z = 0','Data density: 0.1995')],True),
('TRAIN / EVALUATE · reverse the transform and score observed x', [('Observed x = 1','Inverse z = (x−1)/2 = 0'),('log p(x) = log p(z) − log 2','Jacobian magnitude = 2'),('NLL ≈ 1.612 nats','Update transform parameters')],True)])
diagram('vae','VAE: the encoder helps training; the prior starts generation',[
('TRAIN · x ∈ ℝᴰ, latent z ∈ ℝᴹ; M may be smaller than D', [('A: encoder qφ(z|x)','Outputs μ and positive σ'),('B: z = μ + σ ⊙ ε','Independent ε ~ N(0,I)'),('C: decoder pθ(x|z)','Outputs likelihood parameters')],True),
('LOSS · reconstruction through C, prior penalty at A', [('−E log pθ(x|z)','Match the observed x'),('KL(qφ(z|x) || p(z))','Compare latent distributions'),('Update φ and θ','Use reparameterized gradients')],False),
('GENERATE · freeze decoder; encoder is absent', [('Draw z ~ p(z)','No observed x required'),('C: decode z','Distribution parameters'),('Draw x ~ pθ(x|z)','Mean ≠ sampled output')],True)])
diagram('latent','A latent is a distribution during VAE encoding',[
('TOY · one latent coordinate, one reparameterized sample', [('A: μ = 1, σ = 0.5','Variance = 0.25'),('B: ε = −2','z = 1 + 0.5(−2) = 0'),('C: Bernoulli output 0.8','For observed bit x = 1')],True),
('LOSS · one-sample reconstruction estimate + analytic KL', [('Reconstruction: −log 0.8','≈ 0.2231 nats'),('KL to Normal(0,1)','≈ 0.8181 nats'),('Estimated negative ELBO','≈ 1.0413 nats')],False)])
diagram('score','Denoising regression learns a direction; sampling carries a state',[
('TRAIN · fixed σ = 0.5; scalar teaching example', [('Clean x = 2; ε = −1','Noisy input x̃ = 1.5'),('Conditional score target','−ε/σ = 2'),('Predict sθ(x̃)','Squared error; update θ')],True),
('SAMPLE · freeze θ; fresh noise at each Langevin update', [('Current Xₖ','Evaluate score sθ(Xₖ)'),('Xnext = Xₖ + h sθ(Xₖ)','Add √(2h) ξₖ'),('Carry Xnext forward','Repeat; finite-step bias remains')],True),
('ORACLE TRACE · Normal(0,1), score = −X; h = 0.1; ξ = 0', [('X₀ = 2','X₁ = 1.8'),('X₂ = 1.62','X₃ = 1.458'),('Drift-only illustration','Zero noise is not a sampler')],True)])
diagram('gan','GAN: discriminator feedback trains a reusable generator',[
('DISCRIMINATOR UPDATE · generator held fixed', [('Real data x','Generated Gβ(z)'),('Discriminator Dα','Real/fake probabilities'),('Maximize log objective','Update α')],True),
('GENERATOR UPDATE · discriminator held fixed', [('Noise z → Gβ(z)','Differentiable generated data'),('Dα(Gβ(z))','Backpropagate through Dα'),('Improve generator objective','Update β')],True),
('GENERATION · discard discriminator from this path', [('Draw fresh z','Fixed prior'),('Frozen Gβ','One forward transformation'),('Generated sample','No normalized density needed')],True)])
assert math.isclose(.6*.75*.8,.36)
assert math.isclose(.5*(1+.25-1-math.log(.25)),.8181471805599453)
assert math.isclose(-math.log(.8)+.8181471805599453,1.041290731874155)
