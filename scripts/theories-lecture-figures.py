"""Reproduce lecture-guide schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/theories-lecture')
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
diagram('interpolation','Same observed fit, different unseen behavior',[
('OBSERVED · the only two training pairs', [('x = −1; y = −1','Both functions output −1'),('x = 1; y = 1','Both functions output 1'),('Training squared error','Zero for both')],False),
('NEW INPUT · x = 1/2; no observed target here', [('Linear: h₁(x) = x','Prediction = 0.5'),('Cubic: h₂(x) = x³','Prediction = 0.125'),('Which is better?','Depends on the hidden rule')],False),
('TOY PREFERENCE · Ω = w₁² + 4w₃²; λ = 0.1', [('Linear coefficients (1,0)','Penalty = 1; objective = 0.1'),('Cubic coefficients (0,1)','Penalty = 4; objective = 0.4'),('Compare two candidates','Not a global minimizer proof')],False)])
diagram('pipeline','Keep learning, use and measurement distinct',[
('TRAIN · labels are available; weights change', [('x ∈ ℝᵈ → T ∈ ℝᵐ → z ∈ ℝᴷ','Representation → logits'),('Loss + optional penalty','Compare prediction with y'),('Backprop + optimizer','Update w; repeat on batches')],True),
('PREDICT · weights fixed; new label unavailable', [('Fresh input x*','Same feature dimensions'),('Frozen network h_w','No gradient update'),('Class probabilities','Choose prediction')],True),
('DIAGNOSE · different objects, usually after or alongside fitting', [('Hidden states T','Information; structure'),('Weights or distribution Q','BDM; prior-relative KL'),('Local objective geometry','Curvature + prior density')],False),
('TOY ITERATION · h(x) = wx; pair (1,1); step size 1/2', [('w₀ = 0','Gradient = −1'),('w₁ = 0.5','Gradient = −0.5'),('w₂ = 0.75','Inference at x = 2 gives 1.5')],True)])
diagram('information','Retain the task bit; discard the nuisance bit',[
('TOY SOURCE · A and B are independent fair bits; Y = A', [('Input X = (A,B)','Entropy = 2 bits'),('Representation T = A','Entropy = 1 bit'),('Label Y = A','Predictable from T')],True),
('ALTERNATIVE REPRESENTATIONS · same discrete source', [('T = X','I(X;T)=2; I(T;Y)=1'),('T = A','I(X;T)=1; I(T;Y)=1'),('T = constant','I(X;T)=0; I(T;Y)=0')],False),
('IB OBJECTIVE · I(X;T) − 2 I(T;Y); smaller is preferred', [('Keep everything','2 − 2 = 0'),('Keep the relevant bit','1 − 2 = −1'),('Discard everything','0 − 0 = 0')],False)])
diagram('pacbayes','Certify a distribution, not just its mean weights',[
('BOUND INPUTS · P fixed independently of the sample', [('Prior P; learned Q','KL(Q || P) = 10 nats'),('Training risk of Q = 0.05','n = 1000; δ = 0.05'),('Penalty ≈ 0.0926','Risk upper bound ≈ 0.1426')],False),
('RANDOMIZED PREDICTION · Q fixed after training', [('Fresh input x*','Sample predictor h ~ Q'),('Run sampled predictor','Obtain h(x*)'),('Average risk over Q','Guarantee concerns this risk')],True),
('SCOPE · a hypothetical calculation, not a lecture benchmark', [('Sample assumptions','i.i.d.; loss in [0,1]'),('Confidence over S','At least 1 − δ'),('Mean-network accuracy','Not directly certified here')],False)])
diagram('evidence','Evidence is integrated mass around a fit',[
('LAPLACE · local Gaussian approximation at a mode', [('Likelihood × prior density','Peak height'),('Curvature H positive definite','Width ∝ 1 / √det H'),('Height × width × constant','Approximate evidence')],False),
('ONE-DIMENSIONAL TOY · equal likelihood and prior heights', [('Sharper: H = 4','Width factor = 0.5'),('Broader: H = 1','Width factor = 1'),('Broader / sharper = 2','Negative log difference = ln 2')],False),
('CHECK THE ASSUMPTIONS', [('Prior and coordinates','Width alone is insufficient'),('Local shape','Gaussian; nonsingular mode'),('Other modes','May contribute extra mass')],False)])
diagram('lenses','Six lenses inspect different mathematical objects',[
('REPRESENTATION / RANDOMIZED WEIGHTS / LOCAL GEOMETRY', [('1 · Information bottleneck','I(X;T), I(T;Y)'),('2 · PAC-Bayes','Risk(Q), KL(Q || P)'),('3 · Bayesian SGD','Prior-weighted volume')],False),
('TRAINING PROCEDURE / STRUCTURE / DESCRIPTION', [('4 · Double descent','Effective model complexity'),('5 · Grokking','Representation organization'),('6 · Binarized compression','BDM estimate on weights')],False),
('QUESTIONS TO CARRY INTO EACH PAPER', [('What object is measured?','State; weights; procedure?'),('Where does it enter?','Training; prediction; diagnosis?'),('What supports the link?','Theorem; model; experiment?')],False)])
# Original qualitative curves: no empirical coordinates or claimed dataset.
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650" role="img" aria-label="Schematic double descent and grokking with distinct horizontal axes">','<rect width="100%" height="100%" fill="#101c30"/>']
def text(x,y,s,size=20,color='#fff'):
    parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{color}">{escape(s)}</text>')
def path(d,color):
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="4"/>')
text(30,40,'Two phenomena; different axes; schematic shapes',25)
text(30,78,'DOUBLE DESCENT · vary capacity of the training procedure')
path('M90 100 V285 H940','#8299b8')
path('M110 120 C200 220 300 240 390 195 S470 105 490 110 S520 245 920 250','#7dd3fc')
path('M110 160 C220 230 380 270 490 270 H920','#a7f3d0')
text(690,220,'Test error',18,'#7dd3fc');text(700,265,'Training error',18,'#a7f3d0')
text(100,110,'Error',16);text(560,315,'Capacity →',18)
text(335,340,'Near fitting threshold: test error can peak',18)
text(30,390,'GROKKING · fixed task, increasing optimization steps')
path('M90 415 V595 H940','#8299b8')
path('M110 580 C180 580 180 450 260 445 H920','#a7f3d0')
path('M110 580 H560 C650 580 660 450 750 445 H920','#7dd3fc')
text(300,440,'Training accuracy',18,'#a7f3d0');text(735,480,'Validation',18,'#7dd3fc')
text(330,520,'Long delay after fit',18)
text(100,415,'Accuracy',16);text(550,630,'Updates (log scale) →',18)
parts.append('</svg>');(out/'phenomena.svg').write_text('\n'.join(parts))
assert math.isclose(.05+math.sqrt((10+math.log(2*math.sqrt(1000)/.05))/2000),.142581739813,abs_tol=1e-9)
assert .5*.5==.25
print('Generated seven teaching SVGs; certificate arithmetic checked.')
