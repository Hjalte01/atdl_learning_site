"""Reproduce BNN teaching diagrams and Table 3 interval plot; Python stdlib only."""
from pathlib import Path
from html import escape
out=Path(__file__).resolve().parents[1]/'public/figures/bnn'
out.mkdir(parents=True,exist_ok=True)
def text(x,y,s,size=19): return f'<text x="{x}" y="{y}" font-size="{size}">{escape(s)}</text>'
def box(x,y,a,b,c='#ddd9fa',w=270):
 return f'<rect x="{x}" y="{y}" width="{w}" height="90" rx="12" fill="{c}" stroke="#71849a"/>'+text(x+12,y+32,a)+text(x+12,y+65,b,16)
def arrow(x,y,a,b): return f'<path d="M{x} {y} L{a} {b}" stroke="#53677d" stroke-width="3" fill="none" marker-end="url(#a)"/>'
def svg(name,title,desc,body,h=440):
 (out/f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 {h}" role="img" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#53677d"/></marker></defs><rect width="940" height="{h}" rx="18" fill="#f3f6fb"/><g font-family="Arial,sans-serif" fill="#17304b">'+text(28,43,title,25)+body+'</g></svg>')
svg('problem','01 / Count the patterns, or describe the patterns?','Schematic comparison of two block dictionaries with identical frequencies.',box(30,100,'Matrix A: two block types','Each occurs twice','#d4eee7',420)+box(490,100,'Matrix B: two block types','Each occurs twice',w=420)+box(30,255,'Block entropy = 1 bit','Frequencies: 1/2 and 1/2','#d4eee7',420)+box(490,255,'Block entropy = 1 bit','Frequencies: 1/2 and 1/2',w=420)+text(30,390,'BDM also consults each pattern’s CTM cost; identical counts need not mean identical BDM.',18))
svg('architecture','02 / Predict with a BNN; measure its weights separately','Functional architecture and the documented MNIST 32,16 setting.',box(30,95,'A · Input x','MNIST: 10 × 10 → 100','#d4eee7')+arrow(310,140,345,140)+box(350,95,'B · Binary MLP','Hidden widths: 32, 16')+arrow(630,140,665,140)+box(670,95,'C · Class logits','10 outputs → task loss',w=240)+box(30,270,'D · Weight snapshot','Exclude batch normalization','#d4eee7')+arrow(310,315,345,315)+box(350,270,'E · Binary blocks','0/1 encoding; 4 × 4 tiles')+arrow(630,315,665,315)+box(670,270,'F · BDM and entropy','Diagnostics, not loss terms','#ffe6cb',240)+arrow(485,190,170,260)+text(30,417,'B: binarized weights and activations; batch normalization after each hidden layer.',18))
svg('training','03 / Training changes parameters; inference freezes them','Schematic separation of optimizer, diagnostic and prediction paths.',box(30,95,'Train: input + label','A → B → C → cross-entropy','#d4eee7')+arrow(310,140,345,140)+box(350,95,'STE backpropagation','Surrogate for discrete steps')+arrow(630,140,665,140)+box(670,95,'Adam update','Carry weights + moments',w=240)+box(30,265,'Infer: input only','Load selected checkpoint','#d4eee7')+arrow(310,310,345,310)+box(350,265,'A → B → C','Freeze learned parameters')+arrow(630,310,665,310)+box(670,265,'Predicted class','No BDM needed','#ffe6cb',240)+text(30,411,'Training also records D → E → F; no gradient returns from F to the optimizer.',18))
body=''
for k,(x,label,pat) in enumerate([(40,'r₁: checkerboard',[[(i+j)%2 for j in range(4)] for i in range(4)]),(270,'r₂: horizontal bands',[[i%2]*4 for i in range(4)])]):
 body+=text(x,95,label,18)
 for i,row in enumerate(pat):
  for j,v in enumerate(row): body+=f'<rect x="{x+35*j}" y="{115+35*i}" width="33" height="33" fill="{("#d4eee7" if v==0 else "#7158a5")}"/>'
body+=box(520,110,'Four tiles: r₁,r₁,r₁,r₂','Counts 3 and 1; N = 4','#ffe6cb',380)+text(520,250,'H = 0.811 bits per block',22)+text(520,290,'Toy CTM costs: 5 and 9',22)+text(520,330,'BDM = 5 + log₂3 + 9',22)+text(520,370,'= 15.585 (illustrative)',22)+text(30,420,'Tile identities matter to BDM. Tile order is discarded by both count summaries.',18)
svg('blocks','04 / E → F: one partition, two summaries','Computed toy entropy and BDM; CTM values are invented teaching inputs.',body)
svg('numbers','05 / One example takes two different paths','Illustrative classifier output and weight analysis, not trained model outputs.',box(30,95,'A · One training image','Target class y = 3','#d4eee7')+arrow(310,140,345,140)+box(350,95,'B → C · Predict','Probability of y: 0.25')+arrow(630,140,665,140)+box(670,95,'Task loss = 1.386','−ln(0.25); Adam uses this',w=240)+box(30,265,'D · Snapshot weights','w = −0.2, 0, 0.7','#d4eee7')+arrow(310,310,345,310)+box(350,265,'E · Encode signs','bits = 0, 0, 1')+arrow(630,310,665,310)+box(670,265,'F · Aggregate tiles','H / BDM use whole blocks','#ffe6cb',240)+text(30,417,'Three signs illustrate encoding only; the 4 × 4 tiles in Figure 04 supply the metric example.',18))
svg('timeline','06 / From optimization steps to reported correlations','Source preprocessing sequence; no synthetic learning curves.',box(30,95,'Every optimizer step','Record loss, BDM, entropy','#d4eee7')+arrow(310,140,345,140)+box(350,95,'Average within epochs','Validation every 20 steps')+arrow(630,140,665,140)+box(670,95,'Trim patience tail','Omit runs shorter than 5',w=240)+box(30,265,'Per-run transformation','log1p → smooth σ=1 → scale','#d4eee7')+arrow(310,310,345,310)+box(350,265,'Correlate trajectories','Pearson r; Spearman ρ')+arrow(630,310,665,310)+box(670,265,'Bootstrap model runs','95% confidence intervals','#ffe6cb',240)+text(30,417,'200 runs per configuration initially; controls use a fixed 14-epoch window.',18))
rows=[('MNIST',(.74,.79),(.92,.94)),('Fashion-MNIST',(.03,.17),(.30,.46)),('sincos',(.36,.45),(.71,.77)),('UCI HAR',(.68,.77),(.63,.74)),('UCI HAR raw',(.57,.70),(.49,.64))]
body=text(35,80,'Reported Pearson r, 95% bootstrap intervals · source Table 3',19)
for i in range(11):
 x=260+600*i/10
 body+=f'<path d="M{x} 110 V445" stroke="#d1d9e3"/>'+text(x-8,477,str(i/10),14)
for i,(label,a,b) in enumerate(rows):
 y=135+i*65;body+=text(30,y+8,label,19)
 for (lo,hi),dy,color in [(a,-9,'#218c77'),(b,12,'#7158a5')]:
  x1=260+600*lo;x2=260+600*hi
  body+=f'<path d="M{x1} {y+dy} H{x2} M{x1} {y+dy-5} V{y+dy+5} M{x2} {y+dy-5} V{y+dy+5}" stroke="{color}" stroke-width="4"/>'
body+=text(270,515,'Green: block entropy',19)+text(560,515,'Purple: BDM',19)+text(30,560,'Same trained networks for both metrics. Stronger correlation is not higher test accuracy.',18)
svg('results','07 / BDM’s advantage depends on the dataset','Redrawn measured intervals from the supplied May 2026 paper Table 3.',body,590)
print('Generated seven SVGs, including measured Table 3 intervals.')
