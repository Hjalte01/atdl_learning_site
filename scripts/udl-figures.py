"""Reproduce original teaching diagrams; no measured data or model outputs."""
from pathlib import Path
from html import escape
out = Path('public/figures/understanding-deep-learning')
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

diagram('map','A reading map: mechanism before claims',[
('Common foundation', [('Ch. 2–4: prediction', 'Linear → nonlinear → deep'),('Ch. 5–7: learning', 'Likelihood → gradients'),('Ch. 8–9: evaluation', 'Generalization + selection')]),
('Architectures adapted to data', [('Ch. 10–11: images', 'Convolution + residuals'),('Ch. 12: sequences', 'Attention + transformers'),('Ch. 13: graphs', 'Relations + aggregation')]),
('Further reading routes', [('Ch. 14–18: generation', 'Distributions + sampling'),('Ch. 19: acting', 'Reinforcement learning'),('Ch. 20–21: questions', 'Why it works + ethics')])])
diagram('network','One toy classifier: follow the tensor shapes',[
('Learned affine maps; ReLU is fixed', [('Input x: 2 coordinates', 'x = (1, 2)'),('a = Wx + b → ReLU', 'h = (0, 2): 2 coordinates'),('Logit z = vᵀh + c', 'z = 1: scalar')]),
('Continue from the logit; label is needed only for loss', [('Sigmoid probability', 'p = 0.7311: scalar'),('Observed label y = 1', 'Bernoulli loss = 0.3133'),('Backprop + optimizer', 'Update learned weights')])])
diagram('loss','An output distribution determines a training loss',[
('Same prediction, alternative observed label 1', [('Logit z = 1', 'p = 0.7311'),('y = 1: loss = 0.3133', '−log(p)'),('Logit derivative', 'p − y = −0.2689')]),
('Same prediction, alternative observed label 0', [('Logit z = 1', 'p = 0.7311'),('y = 0: loss = 1.3133', '−log(1 − p)'),('Logit derivative', 'p − y = +0.7311')]),
('A different task: fixed-variance Gaussian regression', [('Network predicts mean μ', 'Assume variance s² > 0'),('Negative log-likelihood', '(y − μ)² / (2s²) + const.'),('Least-squares fitting', 'Check noise assumptions')])])
diagram('paths','Training changes weights; inference holds them fixed',[
('TRAIN: repeat for successive batches', [('Inputs + known labels', 'Forward pass; save states'),('Loss → backward pass', 'Chain rule → derivatives'),('Optimizer step', 'Update θ; repeat')]),
('INFER: a new input, no label needed', [('New input x', 'Use learned θ'),('Same prediction network', 'Compute p(y = 1 | x)'),('Probability or decision', 'No optimizer update')])])
import math
z=1.0
states=[]
for k in range(3):
 p=1/(1+math.exp(-z));loss=math.log1p(math.exp(-z))
 states.append((f'Update k = {k}: z = {z:.4f}', f'p = {p:.4f}; loss = {loss:.4f}'))
 z+=0.5*(1-p)
diagram('updates','Repeated updates of the frozen-feature toy classifier',[
('Fixed h = (0, 2), y = 1; update v and c with learning rate 0.1', states),
('Why the logit changes by 0.5(1 − p) per step', [('Δv = 0.1(1 − p)h', 'Δc = 0.1(1 − p)'),('Δz = Δvᵀh + Δc', '||h||² + 1 = 5'),('Recompute probability', 'Then the next gradient')])])
diagram('evaluation','Three data roles answer three different questions',[
('Model development and final assessment', [('Training set', 'Fit each candidate’s θ'),('Validation set', 'Select hyperparameters'),('Reserved test set', 'Assess selected model')]),
('Questions to keep separate', [('Did the optimizer fit?', 'Inspect training objective'),('Which candidate to use?', 'Compare validation metric'),('Does it generalize?', 'Report held-out metric')])])
diagram('attention','Attention routes values using input-dependent weights',[
('One output token; basic single-head mechanism', [('Query q and keys kₘ', 'Dot products → scores'),('Softmax over sources', 'Weights sum to one'),('Mix value vectors vₘ', 'Output o = Σ aₘvₘ')]),
('Scalar teaching example, not learned token embeddings', [('q = 1; keys (0, log 3)', 'Scores (0, log 3)'),('Weights (0.25, 0.75)', 'Values (2, 6)'),('o = 0.25×2 + 0.75×6', 'Contextual output = 5')])])
