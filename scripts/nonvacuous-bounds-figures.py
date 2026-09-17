"""Reproduce PAC-Bayes teaching diagrams and check toy arithmetic; v2 Table 1 data."""
from pathlib import Path
from html import escape
import math
out = Path(__file__).resolve().parents[1] / 'public/figures/nonvacuous-bounds'
out.mkdir(parents=True, exist_ok=True)

def text(x,y,s,size=19):
    return f'<text x="{x}" y="{y}" font-size="{size}">{escape(s)}</text>'
def box(x,y,w,a,b,color='#ddd9fa'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="86" rx="12" fill="{color}" stroke="#71849a"/>'+text(x+14,y+32,a)+text(x+14,y+62,b,16)
def arrow(x,y,a,b):
    return f'<path d="M{x} {y} L{a} {b}" stroke="#53677d" stroke-width="3" fill="none" marker-end="url(#a)"/>'
def svg(name,title,desc,body,h=460):
    (out/f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 {h}" role="img" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#53677d"/></marker></defs><rect width="940" height="{h}" rx="18" fill="#f3f6fb"/><g font-family="Arial,sans-serif" fill="#17304b">'+text(28,43,title,25)+body+'</g></svg>')

def kl(q,p):
    return (q*math.log(q/p) if q else 0)+((1-q)*math.log((1-q)/(1-p)) if q<1 else 0)
def inverse(q,c):
    if q==1 or c==0: return q
    if q==0: return -math.expm1(-c)
    lo,hi=q,1.0
    for _ in range(100):
        mid=(lo+hi)/2
        if mid==hi or mid==lo: break
        if kl(q,mid)>c: hi=mid
        else: lo=mid
    return hi
cost=.5*(.01/.04+.4**2/.04-1+math.log(.04/.01))
loss=math.log1p(math.exp(-1))/math.log(2)
C=(cost+math.log(2000/.05))/1999
q=inverse(.028,math.log(2/.01)/150000)
bound=inverse(q,.1)
print(f'Toy KL={cost:.6f}; loss={loss:.6f}; C={C:.8f}; objective={loss+math.sqrt(C/2):.6f}')
print(f'Certification toy: corrected error={q:.8f}; population bound={bound:.8f}')
assert abs(kl(q,bound)-.1)<1e-12

svg('problem','01 / Same capacity, different generalization','Conceptual contrast; network capacity alone cannot distinguish learning real labels from fitting random labels.',box(30,100,260,'Large model class','Parameters > examples')+arrow(300,143,370,143)+box(380,100,240,'Real training labels','Near-perfect fit','#d4eee7')+arrow(630,143,670,143)+box(680,100,230,'Useful predictions','Can we certify them?','#ffe6cb')+box(380,260,240,'Random training labels','Near-perfect fit','#d4eee7')+arrow(160,196,160,303)+arrow(160,303,370,303)+arrow(630,303,670,303)+box(680,260,230,'Chance test behavior','Memorization is possible','#ffe6cb')+text(30,403,'A useful proof must exploit more than the ability to fit the training sample.',19))
svg('architecture','02 / Predictor + learned weight distribution','T-600 fully connected architecture; tensor dimensions and loss/KL paths are schematic.',box(30,100,245,'A · Images X','B × 784 flattened pixels','#d4eee7')+arrow(285,143,340,143)+box(350,100,255,'C · Dense ReLU','784 → 600; B × 600 activations')+arrow(615,143,670,143)+box(680,100,230,'D · Linear output','600 → 1; B scores')+box(350,270,255,'B · Sample weights v','Q = N(w, diag(s)); v ∈ ℝᵈ')+arrow(475,260,475,196)+box(680,270,230,'E · Logistic loss','Labels + scores → loss','#ffe6cb')+arrow(795,196,795,260)+box(30,270,245,'F · KL(Q ∥ Pλ)','Pλ centered at fixed w₀','#ffe6cb')+text(30,407,'Training: E + complexity from F → update w, s, λ. Prior center w₀ stays fixed.',18)+text(30,449,'Inference: freeze Q → sample v → A, C, D → sign(score). No loss or KL.',18),480)
svg('representation','03 / Independent coordinates, unequal uncertainty','Toy two-weight view: an axis-aligned Gaussian near a prior; ellipses are not hard support boundaries.',arrow(80,335,470,335)+arrow(80,335,80,90)+text(100,100,'Weight 2',17)+text(365,364,'Weight 1',17)+'<ellipse cx="205" cy="220" rx="100" ry="100" fill="#d4eee7" stroke="#218c77" stroke-width="3"/><ellipse cx="315" cy="190" rx="110" ry="38" fill="#ddd9fa" stroke="#8264b0" stroke-width="3"/><circle cx="205" cy="220" r="5" fill="#218c77"/><circle cx="315" cy="190" r="5" fill="#8264b0"/>'+text(130,285,'Prior: w₀, λI',18)+text(265,140,'Posterior: w, diag(s)',18)+box(510,100,390,'Sensitive coordinate → small variance','Too much noise can flip predictions')+box(510,235,390,'KL also charges location and scale','Extreme variances cost more','#ffe6cb')+text(30,415,'Schematic ellipses: Gaussian draws can fall outside them. No loss landscape shown.',18))
svg('training','04 / Train, optimize the distribution, certify','Three different stages with distinct learned and fixed quantities.',box(30,90,270,'1 · Deterministic SGD','Labels → learned wSGD')+arrow(310,133,350,133)+box(360,90,550,'Initialize posterior Q near the SGD solution','Save original random initialization w₀ as prior center','#d4eee7')+arrow(635,185,635,225)+box(360,235,550,'2 · Bound optimization: update w, s, λ','Expected logistic loss + square-root complexity penalty')+box(30,235,270,'Fixed: data and w₀','Fresh Gaussian draw each step','#d4eee7')+arrow(310,278,350,278)+arrow(635,331,635,375)+box(360,385,550,'3 · Freeze Q and compute certificate','Grid prior variance → Monte Carlo → two inverse-KL steps','#ffe6cb')+text(30,526,'Optimizing the smooth objective does not replace final classification-error evaluation.',18),560)
svg('iteration','05 / What carries forward across iterations?','Reparameterized optimization loop; distribution parameters persist and Gaussian noise is refreshed.',box(30,90,265,'Current wₜ, ρₜ, ηₜ','sₜ = exp(2ρₜ); λₜ = exp(2ηₜ)')+arrow(305,133,345,133)+box(355,90,265,'Sample ξₜ ~ N(0, I)','vₜ = wₜ + exp(ρₜ) ⊙ ξₜ','#d4eee7')+arrow(630,133,670,133)+box(680,90,230,'Network forward','Full training sample')+box(680,270,230,'Loss + KL penalty','Backpropagate both paths','#ffe6cb')+arrow(795,186,795,260)+box(30,270,590,'Next wₜ₊₁, ρₜ₊₁, ηₜ₊₁ → repeat with fresh noise','Reported experiments: RMSprop; Algorithm 1 illustrates vanilla SGD')+arrow(670,313,630,313)+arrow(160,260,160,186)+text(30,410,'Keep w₀ fixed. The sampled vₜ is not the next posterior mean.',19)+text(30,448,'Positive scales from exponentiation; the prior search additionally requires λ < c.',18),480)
svg('numbers','06 / One sampled weight, two gradient paths','Computed scalar illustration with a fixed prior; not MNIST or a final certificate.',box(30,95,260,'A / B · x = 2; ξ = 1','w = 0.4; √s = 0.1 → v = 0.5','#d4eee7')+arrow(300,138,350,138)+box(360,95,260,'C / D · Score fᵥ(x) = 1','Predict +1; label y = +1')+arrow(630,138,670,138)+box(680,95,230,'E · Logistic loss',f'{loss:.3f}; zero mistakes','#ffe6cb')+box(30,275,260,'F · Gaussian KL',f'w₀ = 0; λ = 0.04 → {cost:.3f} nats','#ffe6cb')+arrow(300,318,350,318)+box(360,275,260,'Complexity penalty',f'm = 2000; δ = 0.05 → {math.sqrt(C/2):.4f}')+arrow(630,318,670,318)+box(680,275,230,'Sampled objective',f'{loss:.3f} + {math.sqrt(C/2):.4f} ≈ {loss+math.sqrt(C/2):.3f}')+arrow(795,191,795,265)+text(30,420,'Toy dataset: 2000 identical observations. One draw estimates expected logistic loss.',18))
svg('certificate','07 / Account for two sources of uncertainty','Toy nested inverse-KL calculation using the source Monte Carlo confidence settings.',box(30,100,265,'Freeze posterior Q','n = 150,000 network draws')+arrow(305,143,345,143)+box(355,100,265,'Average training error','Toy Monte Carlo value: 0.028','#d4eee7')+arrow(630,143,670,143)+box(680,100,230,'First inverse KL',f'δ′ = 0.01 → {q:.5f}','#ffe6cb')+box(30,285,265,'Grid prior + analytic KL','Toy final budget BRE = 0.1','#d4eee7')+arrow(305,328,345,328)+box(355,285,265,'Second inverse KL',f'Population error ≤ {bound:.4f}','#ffe6cb')+arrow(795,196,795,235)+arrow(795,235,490,235)+arrow(490,235,490,275)+text(655,330,'δ + δ′ = 0.035',19)+text(655,366,'Confidence: 96.5%',19)+text(30,427,'n counts independent network draws, not n × m independent predictions.',18))
rows=[('600',3.4,16.1),('1200',3.5,17.9),('300²',3.4,17.0),('600²',3.3,18.6),('1200²',3.5,22.3),('600³',3.2,20.1)]
body=text(30,85,'Binary MNIST · 55,000 training / 10,000 test images · true labels',18)
for i,(label,test,bnd) in enumerate(rows):
    y=130+i*63
    body+=text(32,y+13,'T-'+label,18)
    body+=f'<rect x="155" y="{y-7}" width="{test*25}" height="15" fill="#218c77"/>'+text(163+test*25,y+6,f'{test:.1f}%',15)
    body+=f'<rect x="155" y="{y+13}" width="{bnd*25}" height="15" fill="#8264b0"/>'+text(163+bnd*25,y+26,f'{bnd:.1f}%',15)
body+=text(30,553,'Green: SNN test-error upper estimate   Purple: PAC-Bayes population bound',18)+text(30,587,'Source: v2 Table 1 / §4.4. Certificate confidence 0.965. Lower is better.',18)+text(30,621,'Random-label 1.352 uses a different definition and is excluded from these bars.',17)
svg('results','08 / A useful certificate can still be loose', 'Measured Table 1 values; six SNN test-error upper estimates and six PAC-Bayes bounds in percent.',body,650)
