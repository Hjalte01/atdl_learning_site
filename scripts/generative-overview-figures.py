"""Reproduce original teaching diagrams; no measured data or model outputs."""
from pathlib import Path
from html import escape
out = Path('public/figures/generative-overview')
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
            if j<2 and not (name == "map" and i == 0 or name == "families" and i == 3): parts.append(f'<path d="M{x+278},{y+49} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    parts.append('</svg>')
    (out/f'{name}.svg').write_text('\n'.join(parts))
diagram('map','Three questions organize the reading',[('1 · How is a distribution learned?', [('GAN: adversarial game',),('VAE: latent variables',),('DDPM: denoising',)]),('2 · How is generation accelerated?', [('Discrete reverse steps',),('Continuous trajectories',),('Learn transport velocity',)]),('3 · What should be generated?', [('Unconditional sample',),('Condition on context',),('Edit or predict structure',)])])
diagram('families','Functional mechanisms: training and generation',[('GAN training: discriminator sees real examples too', [('Noise z',),('Generator G(z)',),('Discriminator feedback',)]),('VAE training: reconstruction plus prior regularization', [('Data x',),('Encoder → sample z',),('Decoder reconstructs x',)]),('DDPM training: target noise is known', [('Data x + drawn noise',),('Noisy state and time',),('Predict added noise',)]),('Generation: use the learned machinery', [('GAN: z → generator',),('VAE: prior z → decoder',),('DDPM: repeated denoise',)])])
diagram('paths','Training changes weights; sampling changes a state',[('Training: repeat over examples, times and noise draws', [('Construct noisy input',),('Predict noise / velocity',),('Loss → update weights',)]),('Inference: weights stay fixed', [('Initial noise state',),('Model + solver update',),('Next state → repeat',)])])
diagram('numeric','A scalar flow example, with noise at time zero',[('Training pair: noise z = 2, data x = 6', [('t = 0: state 2',),('t = 0.5: state 4',),('t = 1: state 6',)]),('Toy target velocity is x − z = 4', [('Current state: 2',),('Step: 2 + 0.5 × 4',),('Next state: 4',)]),('This is arithmetic, not a trained generator', [('Known pair for teaching',),('Learned field can differ',),('Large steps can err',)])])
diagram('conditioning','Separate context from the evolving generated state',[('Unconditional generation', [('Noise',),('Learned generator',),('New sample',)]),('Conditional generation or editing', [('Noise + fixed context',),('Conditioned updates',),('Image fitting context',)]),('Structured prediction', [('Observed image',),('Model + task guidance',),('Mask or explanation',)])])
diagram('route','A suggested route through the six existing guides',[('Understand transport, then training efficiency', [('This overview',),('Mean Flows · paper 2',),('REPA · paper 1',)]),('Understand conditioning and editing', [('This overview',),('FLUX Kontext · paper 5',),('Qwen-Image · paper 6',)]),('Understand structured outputs and explanations', [('This overview',),('DiffAtlas · paper 4',),('DiME · paper 3',)])])
