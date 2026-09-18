"""Reproduce lecture-guide schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/generative-lecture-2')
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
diagram('map','One generator combines several independent choices',[
('LEARNED TARGET · what does the network predict?', [('Noise ε','DDPM: slides 2–3'),('Score ∇ log p','Score models: slides 7–12'),('Velocity v','Flow matching: slide 17')],False),
('SAMPLING · how does the state move?', [('Stochastic updates','Fresh noise at steps'),('Deterministic integration','ODE or deterministic DDIM'),('Endpoint prediction','Consistency: slide 13')],False),
('REPRESENTATION · what object evolves?', [('Pixel values','Full image tensor'),('Latent / wavelet features','Encode or transform'),('Discrete visual tokens','MaskGIT: slide 24')],False),
('CONDITIONING · where does outside information enter?', [('Attention','Route condition information'),('FiLM','Scale and shift channels'),('Guidance','Combine sampling predictions')],False)])
diagram('ddpm','DDPM: learning the correction, then repeating it',[
('TRAIN · clean x₀, sampled t and known noise ε; schedule is fixed', [('A · Corruption','xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε'),('B · Denoiser εθ(xₜ,t)','U-Net receives image + time'),('C · Squared noise loss','Compare to ε; update θ')],True),
('SAMPLE · θ fixed; start xT ~ N(0,I); t decreases', [('Current state xₜ','No clean target at inference'),('Same denoiser → mean μθ','Add σₜ z at intermediate steps'),('Carry xₜ₋₁ forward','Repeat until clean output')],True),
('TOY TRACE · one scalar training example', [('x₀ = 2, ᾱₜ = 0.64, ε = −1','A gives xₜ = 1'),('B predicts −0.5','C gives loss = 0.25'),('Sampling mean ≈ 1.304','αₜ = 0.8; βₜ = 0.2')],True)])
diagram('score','Same score, different sampler drift',[
('REVERSE SDE · backward duration h > 0', [('xₜ + time t','Score sθ ≈ ∇ log pₜ'),('Subtract (f − g²sθ)h','Add fresh g√h z'),('xₜ₋h','Repeat with new noise')],True),
('PROBABILITY-FLOW ODE · backward duration h > 0', [('xₜ + time t','Same score function'),('Subtract (f − ½g²sθ)h','No fresh sampling noise'),('xₜ₋h','Repeat deterministic updates')],True),
('TOY · xₜ = 2; f = 0; g = 1; sθ = −0.5; h = 0.1', [('Reverse SDE','1.95 + √0.1 z'),('Probability-flow ODE','1.975'),('Ideal shared marginals','Not identical trajectories')],False)])
diagram('flow','Flow matching: train on directions, sample with a solver',[
('TRAIN · slide 17 convention: time 0 = noise; time 1 = data', [('Pair x₀ = −2, x₁ = 2','Draw t = 0.25'),('Interpolate xₜ = −1','Target u = x₁ − x₀ = 4'),('Fit vθ(xₜ,t) to u','Update θ using squared error')],True),
('SAMPLE · θ fixed; data endpoint absent; h = 0.25', [('t = 0; x = −2','Toy v = 4'),('t = 0.25; x = −1','x ← x + h v'),('t = 0.5; x = 0','Call field again')],True),
('CONTINUE · a constant perfect toy field; real learned fields can curve', [('t = 0.5; x = 0','Carry current state'),('t = 0.75; x = 1','Call field again'),('t = 1; x = 2','Stop at data boundary')],True)])
diagram('representation','Choose the evolving representation explicitly',[
('PIXELS · model the image coordinates directly', [('Training image x','Shape H × W × C'),('Noise + learned denoising','Same image coordinate space'),('Final pixel state','Image output')],True),
('WAVELETS · invertible rearrangement, not necessarily compression', [('Image','Apply fixed DWT'),('Frequency bands','Generate band coefficients'),('Apply inverse DWT','Reconstructed image')],True),
('LEARNED LATENT · components trained in separate stages', [('Encoder E(x) = z','Shape h × w × c'),('Latent denoiser','Fit on noisy encoded images'),('Decoder D(z)','Reconstruct from clean latent')],False),
('LATENT GENERATION · no source-image encoder needed', [('Draw latent noise zT','Condition encoded separately'),('Repeat frozen denoiser','Carry latent toward z₀'),('Decode D(z₀)','Final image')],True)])
diagram('conditioning','Three ways conditions affect the computation',[
('ATTENTION · route values inside the network', [('Scaled query-key logits','(0, ln 3)'),('Row softmax weights','(1/4, 3/4)'),('Values (2, 6)','Weighted output = 5')],True),
('FiLM · condition produces channel scale and shift', [('Condition → γ = 2, β = −1','Feature F = 3'),('Apply γF + β','2 × 3 − 1'),('Modulated feature = 5','Broadcast over spatial map')],True),
('CLASSIFIER-FREE GUIDANCE · combine predictions for the sampler', [('Same xₜ and time t','s_uncond = 1; s_cond = 3'),('Scale γ = 2','1 + 2 × (3 − 1)'),('Guided prediction = 5','Pass to the update rule')],True)])
diagram('metrics','Different metrics ask different questions',[
('CLASS CONFIDENCE AND FEATURE DISTRIBUTIONS · slides 32 and 34', [('Inception Score ↑','Confident + varied classes'),('FID ↓','Mean / covariance mismatch'),('CMMD ↓','Kernel discrepancy in CLIP')],False),
('COVERAGE AND UTILITY · slide 33', [('Precision','Are generated samples near real?'),('Recall','Does generation cover real?'),('Task scores / human tests','Does it serve the intended use?')],False),
('TOY 1D FID · real mean 0, SD 1; generated mean 1, SD 2', [('Mean penalty','(0 − 1)² = 1'),('Spread penalty','(1 − 2)² = 1'),('Sum = 2','Calculated, not benchmark data')],False)])
assert math.isclose(.8*2+.6*(-1),1)
assert math.isclose((1-.2/.6*(-.5))/math.sqrt(.8),1.304372986874877)
assert math.isclose(.25*2+.75*6,5)
assert math.isclose(-1+.25*4,0)
print('Generated seven SVGs; toy arithmetic passed.')
