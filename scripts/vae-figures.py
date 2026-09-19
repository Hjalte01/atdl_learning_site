"""Reproduce VAE teaching figures and verify scalar arithmetic.

Optionally set POPPLER_PDFTOPPM to regenerate the unaltered source page.
"""
from pathlib import Path
from html import escape
import math
import os
import subprocess
out=Path('public/figures/vaes');out.mkdir(parents=True,exist_ok=True)
def diagram(name,title,rows):
    height=110+len(rows)*125
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-label="{escape(title)}">','<rect width="100%" height="100%" fill="#101c30"/>','<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>',f'<text x="30" y="42" fill="#fff" font-family="sans-serif" font-size="25">{escape(title)}</text>']
    for i,(label,boxes,arrows) in enumerate(rows):
        y=85+i*125
        parts.append(f'<text x="30" y="{y}" fill="#c5d5ee" font-family="sans-serif" font-size="18">{escape(label)}</text>')
        for j,lines in enumerate(boxes):
            x=30+j*325
            parts.append(f'<rect x="{x}" y="{y+15}" width="290" height="68" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
            for k,line in enumerate(lines):parts.append(f'<text x="{x+12}" y="{y+42+k*24}" fill="#fff" font-family="sans-serif" font-size="17">{escape(line)}</text>')
            if arrows and j<len(boxes)-1:parts.append(f'<path d="M{x+294},{y+49} H{x+319}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#arrow)"/>')
    (out/f'{name}.svg').write_text('\n'.join(parts+['</svg>']))
diagram('architecture','A probabilistic encoder teaches a probabilistic generator',[
('INFERENCE · observed x enters a shared recognition network', [('A: observation x','D observed coordinates'),('B: encoder qφ(z | x)','tanh hidden layer'),('Two J-dimensional heads','Mean μ; log variance ℓ')],True),
('SAMPLING AND DECODING · ε ~ N(0,I), independent of learned weights', [('C: z = μ + exp(ℓ/2) ⊙ ε','J latent coordinates'),('D: decoder pθ(x | z)','tanh hidden layer'),('Observation parameters','D probabilities, or μx and ℓx')],True),
('OBJECTIVE · compare to A, and compare B to the fixed prior', [('D + observed x','Reconstruction log likelihood'),('B + prior p(z) = N(0,I)','Analytic prior KL'),('E: log likelihood − KL','Gradients update φ and θ')],False)])
diagram('objective','Two KL divergences, two different jobs',[
('COMPUTABLE ELBO · maximize this expected objective at E', [('Expected reconstruction','E q [log pθ(x | z)]'),('Subtract prior KL','KL(qφ(z | x) || p(z))'),('ELBO = reconstruction − KL','Teaching example: −2.4')],True),
('EXACT IDENTITY · the gap is usually unavailable', [('ELBO','Example: −2.4'),('Add posterior KL','KL(qφ(z | x) || pθ(z | x))'),('Exact log evidence','Example: −2.1; gap 0.3')],True),
('BOUND VS ESTIMATE', [('Expected ELBO ≤ log pθ(x)','Nonnegative posterior KL'),('One sampled estimate','Can fluctuate above evidence'),('Zero prior KL ≠ tight bound','Prior is not the posterior')],False)])
diagram('reparameterization','Keep the random draw fixed while taking the derivative',[
('FORWARD · box C, one coordinate', [('Draw ε ~ N(0,1)','No learned noise parameters'),('Encoder gives μ and ℓ','σ = exp(ℓ/2)'),('Combine z = μ + σε','Same Gaussian sample law')],True),
('BACKWARD · reconstruction derivative arriving from decoder D', [('∂R/∂z','Decoder input derivative'),('To mean head','∂R/∂μ = ∂R/∂z'),('To log-variance head','∂R/∂ℓ = (∂R/∂z) σε/2')],False),
('NUMERICAL EXAMPLE · μ = 0.5, σ = 0.5, ε = 1', [('Forward: z = 1','Decoder: y ≈ 0.7311'),('Reconstruction ∂R/∂z','≈ −0.2689'),('Reconstruction ∂R/∂ℓ','≈ −0.0672; then add KL')],True)])
diagram('paths','Three tasks; only training changes weights',[
('TRAIN · both networks learn', [('Observed x → B','μ, ℓ and random ε → C'),('Sample z → D','Evaluate observed x'),('E: reconstruction + KL','Minimize negative ELBO')],True),
('RECONSTRUCT · keep φ and θ fixed', [('Observed x → B','Infer qφ(z | x)'),('Draw z from qφ','Or choose mean explicitly'),('D: distribution over x','Display mean or sample x')],True),
('GENERATE · keep θ fixed; no encoder needed', [('No input image','Draw z from prior N(0,I)'),('D: decode z','Observation distribution'),('Draw x from pθ(x | z)','Mean display is a summary')],True)])
diagram('loop','AEVB repeats stochastic learning, not iterative inference',[
('ITERATION k · weights and optimizer state enter from the previous step', [('Replace minibatch','M observations from N'),('Replace random noise','L draws per observation'),('B → C → D → E','Compute sampled objective')],True),
('JOINT UPDATE · gradient ascent on ELBO, or descent on its negative', [('Backpropagate to φ and θ','Do not detach latent z'),('Apply optimizer update','Paper uses Adagrad'),('Carry new weights forward','Return to the next minibatch')],True),
('REPORTED SETTINGS · Algorithm 1 and §5', [('M = 100','L = 1 per observation'),('Exact Gaussian prior KL','Only reconstruction sampled'),('Weight prior in experiments','Approximate MAP training')],False)])
diagram('worked','One binary observation through the same five boxes',[
('FORWARD · constructed scalar example, not a trained MLP', [('A: x = 1; B: μ = σ = 0.5','C: ε = 1 gives z = 1'),('D: w = 1 and b = 0','y = sigmoid(wz+b) ≈ 0.7311'),('E: R ≈ 0.3133; K ≈ 0.4431','Negative ELBO ≈ 0.7564')],True),
('BACKWARD · reconstruction plus prior KL', [('Decoder input derivative','∂R/∂z ≈ −0.2689'),('Mean gradient ≈ 0.2311','Log-var gradient ≈ −0.4422'),('Decoder weight and bias','Both gradients ≈ −0.2689')],True),
('ILLUSTRATIVE DESCENT · η = 0.1; treat encoder outputs as parameters', [('μ new ≈ 0.4769','ℓ new ≈ −1.3421'),('w new ≈ 1.0269','b new ≈ 0.0269'),('Real encoder: chain rule','Update shared neural weights')],False)])
# An actual density plot for the representation close-up.
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="470" viewBox="0 0 1000 470" role="img" aria-label="Prior N(0,1) and toy encoder N(1,0.25) density curves"><rect width="100%" height="100%" fill="#101c30"/><g fill="white" font-family="sans-serif"><text x="30" y="42" font-size="25">An encoder predicts a distribution, not one fixed code</text><text x="30" y="78" font-size="18">Teaching example: prior N(0,1); posterior approximation N(1,0.25)</text>']
for mu,sigma,color in [(0,1,'#69c8ec'),(1,.5,'#c2a7ff')]:
    points=[]
    for i in range(501):
        z=-3+6*i/500;density=math.exp(-.5*((z-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        points.append(f'{90+(z+3)*135:.2f},{340-density*260:.2f}')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4"/>')
parts.append('<path d="M90,110 V340 H900" stroke="#c5d5ee" fill="none"/><text x="32" y="112" font-size="16">density</text>')
for z in range(-3,4):parts.append(f'<text x="{85+(z+3)*135}" y="365" font-size="17">{z}</text>')
parts.extend(['<text x="920" y="345" font-size="18">z</text><text x="155" y="185" fill="#69c8ec" font-size="20">prior</text><text x="670" y="150" fill="#c2a7ff" font-size="20">q(z | x)</text>', '<text x="50" y="405" font-size="19">Encoder mean 1, standard deviation 0.5: z = 1 + 0.5ε</text><text x="50" y="441" font-size="19">ε = −1 → z = 0.5          ε = 0 → z = 1          ε = 1 → z = 1.5</text></g></svg>'])
(out/'representation.svg').write_text('\n'.join(parts))
y=1/(1+math.exp(-1));K=.5*(.25+.25-1-math.log(.25));R=-math.log(y)
assert round(K+R,4)==.7564
assert round(y-1+.5,4)==.2311
assert round((y-1)*.25+(.25-1)/2,4)==-.4422
assert round(math.log(.25)-.1*((y-1)*.25+(.25-1)/2),4)==-1.3421
if os.environ.get('POPPLER_PDFTOPPM'):
    subprocess.run([os.environ['POPPLER_PDFTOPPM'],'-f','7','-singlefile','-scale-to','1500','-png','public/papers/auto-encoding-variational-bayes.pdf',str(out/'source-page-7')],check=True)
print('Seven VAE teaching figures generated; sampled loss and gradients verified.')
