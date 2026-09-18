"""Reproduce original teaching diagrams; no measured data or model outputs."""
from pathlib import Path
from html import escape
out = Path('public/figures/representation-overview')
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
            if j < len(boxes)-1: parts.append(f'<path d="M{x+278},{y+49} H{x+305}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#a)"/>')
    parts.append('</svg>')
    (out/f'{name}.svg').write_text('\n'.join(parts))
diagram('map','From geometry to a usable representation',[
('Foundations: find and describe structure', [('Ch. 1: low-dimensionality',),('Ch. 2: linear models',),('Ch. 3: denoise data',)]),
('Representation: preserve useful differences', [('Ch. 4: coding-rate gain',),('Ch. 5: unrolled networks',),('Ch. 6: consistent codes',)]),
('Use, evaluate, question', [('Ch. 7: inference',),('Ch. 8: implementations',),('Ch. 9: open directions',)])])
diagram('pca','A complete linear encoder–decoder: D = 2, d = 1',[
('A fixed teaching basis U = (1, 1) / √2', [('Input x = (3, 1)', 'Two ambient coordinates'),('Encode z = Uᵀx = 2√2', 'One latent coordinate'),('Decode Uz = (2, 2)', 'Back in two dimensions')]),
('The signal assumption decides what gets discarded', [('Reconstruction (2, 2)', 'Along the diagonal line'),('Residual x − Uz = (1, −1)', 'Perpendicular to the line'),('Squared error = 2', 'Not always pure noise')])])
diagram('paths','Two loops with different changing quantities',[
('TRAIN: clean target x is available', [('Draw x, t > 0 and noise', 'Construct xₜ = x + tε'),('Denoiser Dθ(xₜ, t)', 'Predict clean x'),('Squared loss → weights', 'Repeat over batches')]),
('SAMPLE: no clean target; learned θ stays fixed', [('Initial noisy state', 'Choose sampler/schedule'),('Denoiser + solver', 'Update the state'),('Next noise level', 'Repeat to obtain output')])])
diagram('rate','Coding-rate reduction rewards group structure',[
('Unit-norm samples; d = 2, N = 4, ε = 1; natural logs', [('Group A: ±(1, 0)', 'Group B: ±(0, 1)'),('Whole R = 0.6931', 'Within Rᶜ = 0.5493'),('Subtract: ΔR = 0.1438', 'Groups on distinct axes')]),
('Keep norms fixed; move both groups onto the same axis', [('Group A: ±(1, 0)', 'Group B: ±(1, 0)'),('Whole R = 0.5493', 'Within Rᶜ = 0.5493'),('Subtract: ΔR = 0', 'Groups overlap')])])
diagram('unroll','Depth transforms features; training learns parameters',[
('Functional CRATE encoder: n tokens, feature width d', [('Input → embeddings', 'Z⁰: d × n'),('L repeated blocks', 'Zˡ → Zˡ⁺¹: d × n'),('Final representation', 'Task-specific head')]),
('Inside a block: functional view, normalization omitted', [('Current features Zˡ',),('Subspace attention', 'MSSA: compression'),('Sparse-coding stage', 'ISTA-inspired update')]),
('Outer training loop is separate from network depth', [('A batch + task target',),('Forward pass → loss',),('Backprop → parameters', 'Then another batch')])])
diagram('loop','A round trip checks the representation',[
('Encode, then decode: Eq. (6.2.14), schematic', [('Data X', 'D × N'),('Encoder fθ → Z', 'd × N'),('Decoder gη → X̂', 'D × N')]),
('Continue from X̂; reuse the same encoder fθ', [('Reconstructed X̂',),('Encoder fθ → Ẑ', 'd × N'),('Compare Z with Ẑ', 'Feature-space feedback')]),
('Complementary roles in the training game', [('Encoder exposes mismatch',),('Decoder reduces mismatch',),('Also structure the code', 'Agreement alone can fail')])])
# Only this diagram contains measured source values; no inferred error bars.
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="410" viewBox="0 0 960 410" role="img" aria-label="Selected Table 8.5 ImageNet-1K linear probe results">', '<rect width="960" height="410" fill="#101c30"/>']
def text(x,y,s,size=20,color='#fff'):
    parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="sans-serif" font-size="{size}">{escape(s)}</text>')
text(30,42,'Reported evidence: a parameter–accuracy trade-off',25)
text(30,78,'Table 8.5 · ImageNet-1K linear-probe accuracy · March 1, 2026 copy',18,'#c5d5ee')
for i,(name,value,params,color) in enumerate([('CRATE-S',69.2,'13.12M','#57c4b7'),('ViT-S',72.4,'22.05M','#b69bf5')]):
    y=130+i*95
    text(30,y+23,name); text(30,y+48,params+' parameters',16,'#c5d5ee')
    parts.append(f'<rect x="240" y="{y}" width="{value*6}" height="40" fill="{color}" rx="4"/>')
    text(250+value*6,y+27,f'{value}%')
for value in [0,25,50,75,100]:
    x=240+value*6
    parts.append(f'<path d="M{x},285 v8" stroke="#c5d5ee"/>')
    text(x-8,318,str(value),16,'#c5d5ee')
parts.append('<path d="M240,285 H840" stroke="#c5d5ee"/>')
text(30,362,'150 epochs supervised pretraining · patch size 16 · no intervals supplied',18)
text(30,390,'Selected comparison only. Parameter count does not measure latency.',18,'#c5d5ee')
parts.append('</svg>')
(out/'evidence.svg').write_text('\n'.join(parts))
