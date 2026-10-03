"""Reproduce SAM 3 conference-version teaching diagrams from repository root."""
from pathlib import Path
from html import escape
from math import log, isclose, sqrt
out = Path('public/figures/sam3'); out.mkdir(parents=True, exist_ok=True)
BG='#101c30'; WHITE='#eef4ff'; BLUE='#258ca6'; PURPLE='#7960b0'; GREEN='#339774'
def start(title,h):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{h}" viewBox="0 0 960 {h}" role="img" aria-label="{escape(title)}">', f'<rect width="960" height="{h}" fill="{BG}"/>', '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c5d5ee"/></marker></defs>']
def text(p,x,y,s,size=18,color=WHITE):
 p.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{color}">{escape(s)}</text>')
def finish(name,p): (out/f'{name}.svg').write_text('\n'.join(p+['</svg>'])+'\n')
def flow(name,title,rows):
 p=start(title,110+len(rows)*150);text(p,28,40,title,24)
 for i,(label,boxes) in enumerate(rows):
  y=82+i*150;text(p,28,y,label,18,'#c5d5ee')
  for j,lines in enumerate(boxes):
   x=28+j*312;p.append(f'<rect x="{x}" y="{y+16}" width="278" height="94" rx="10" fill="{["#174c63","#493a75","#245546"][j]}" stroke="#8299b8"/>')
   for k,line in enumerate(lines):text(p,x+12,y+43+k*23,line,17)
   if j<len(boxes)-1 and name!='representation' and not (name=='architecture' and i==0) and not (name=='presence' and i==0):p.append(f'<path d="M{x+281},{y+62} H{x+307}" stroke="#c5d5ee" stroke-width="2" marker-end="url(#arrow)"/>')
 finish(name,p)
p=start('A point selects an instance; a concept finds a set',430);text(p,28,40,'A point selects an instance; a concept finds a set',24)
for row,(label,selected) in enumerate([('PVS: click one apple',[0]),('PCS: “red apple”',[0,1,3])]):
 y=105+row*155;text(p,28,y,label,21)
 for j in range(4):
  x=345+j*140;p.append(f'<circle cx="{x}" cy="{y+25}" r="38" fill="{ "#ca5364" if j!=2 else "#d8c148"}"/>');p.append(f'<path d="M{x},{y-14} q5,-18 18,-15" stroke="#57bb85" stroke-width="5" fill="none"/>')
  if j in selected:p.append(f'<rect x="{x-48}" y="{y-27}" width="96" height="103" rx="13" fill="none" stroke="#63d7ed" stroke-width="4"/>')
  if row==0 and j==0:p.append(f'<circle cx="{x}" cy="{y+25}" r="7" fill="white"/>')
text(p,28,405,'Schematic apples and mask outlines; no SAM model predictions.',18);finish('problem',p)
flow('architecture','Shared visual features, distinct detection and tracking paths',[
('A · INPUTS: independent branches meet at B', [('Image I → PE vision encoder','Unconditioned features F'),('Text → PE text encoder','Exemplar → ROI + label + box'),('Prompt tokens P','Text and/or exemplars')]),
('B → C → D · IMAGE DETECTOR', [('B: F cross-attends to P','6-block fusion encoder'),('C: 6-block query decoder','200 queries + presence token'),('D: boxes, masks, scores','Semantic map + instance set')]),
('VIDEO BRANCH: starts from A features, not conditioned B features', [('A: F + per-object memory','Memory attention'),('SAM 2-style mask decoder','Propagated masks with IDs'),('Associate with D detections','Spawn / confirm / re-prompt')]),
('RECURRENT STATE: match/update → future memory attention', [('Mask + A features','Memory encoding per object'),('Keep confident references','Spatial memory + pointers'),('Next frame reuses identity','Detector can discover arrivals')])])
flow('representation','A prompt token is different from an object query',[
('A AND B · INPUT REPRESENTATIONS', [('Vision: F has N × d channels','One cached grid per frame'),('Text + exemplar tokens','Prompt has L × d channels'),('B: conditioned grid N × d','Same positions, new context')]),
('C · OUTPUT SLOTS, NOT USER-SUPPLIED OBJECTS', [('Q = 200 learned queries','Decoder states Q × d'),('One shared presence token','Predicts concept existence'),('Queries give boxes + scores','D maps features to masks')]),
('D · INSTANCE MASKS AND SEMANTIC MAP', [('Instance: per-object masks','H × W pixels per mask'),('Semantic: matching pixels','No instance identity required'),('Video: per-object memory','Tracks ID through time')])])
flow('presence','Recognition gates every local candidate score',[
('C → D · SAME PRESENCE FACTOR FOR ALL QUERIES', [('Global token → MLP','r = p(concept present)'),('Query i → local score','aᵢ = p(match | present)'),('Final score sᵢ = r × aᵢ','Retain scores above .5')]),
('TOY POSITIVE IMAGE: r = .8', [('Local scores: .9, .7, .4','Three illustrative queries'),('Final: .72, .56, .32','Two candidates pass .5'),('r falls to .2?','All scores fall below .5')]),
('LEARNING: ABSENT CONCEPTS TEACH THE GLOBAL BRANCH', [('Presence BCE: positives','and hard negatives'),('Per-query classification','only on positive concepts'),('Local unmatched queries','still get negative labels')])])
flow('training','Training changes weights; inference carries prompts and memory',[
('STAGES 1 → 2 → 3 · IMAGE AND LANGUAGE LEARNING', [('1: aligned PE encoders','5.4B image-text pairs'),('2: detector + encoders','~95k steps; broad mask data'),('3: high-quality refinement','~5k; presence + interaction')]),
('STAGE 4 · VIDEO TRACKER LEARNING', [('Freeze shared PE backbone','Keep image grounding'),('Learn tracker + memory path','~190k steps'),('Longer temporal context','60k additional steps')]),
('INFERENCE · NO OPTIMIZER OR GROUND-TRUTH LOSS', [('Cache frame features F','Apply text/exemplar prompt'),('Detector gives concept set','Clicks refine one instance'),('Video state: IDs + memory','New frame repeats inference')]),
('DATA ENGINE · DIFFERENT LOOP FROM VIDEO INFERENCE', [('Mine media + noun phrases','Propose masks + negatives'),('Verify quality + exhaustivity','Humans correct failures'),('Train next model on labels','Not online user fine-tuning')])])
flow('numeric','Trace one invented example through A → B → C → D',[
('A → B · REPRESENTATIONS', [('Image contains 2 red apples','PE caches image feature grid'),('Text: “red apple”','Fusion conditions the grid'),('No numeric encoder claimed','Head outputs assumed below')]),
('C → D · SCORING', [('Presence r = .8','Local a = [.9, .7, .4]'),('s = [.72, .56, .32]','Threshold > .5 keeps 2'),('Masks overlap two apples','Mask quality is separate')]),
('TRAINING ARROW AT C · POSITIVE PRESENCE LABEL y = 1', [('BCE = −ln(.8) = .2231','Stage 3 weight: 20'),('Contribution ≈ 4.4629','Only the presence term'),('dBCE/dlogit = .8 − 1','Gradient descent raises r')])])
p=start('A masklet score adds evidence across frames',490);text(p,28,40,'A masklet score adds evidence across frames',24);text(p,28,80,'Toy sequence: detector match +1, no match −1; cumulative MDS',18)
deltas=[1,1,-1,-1,-1,1,1]; scores=[];v=0
for d in deltas:v+=d;scores.append(v)
p.append('<path d="M130 160 V360 H875 M130 295 H875" fill="none" stroke="#8299b8"/>')
for tick in [-1,0,1,2]:text(p,95,300-tick*55,str(tick),16)
pts=[]
for i,(d,s) in enumerate(zip(deltas,scores)):
 x=185+i*105;y=295-s*55;pts.append(f'{x},{y}');p.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#63d7ed"/>');text(p,x-9,385,str(i+1),16);text(p,x-12,135,'+1' if d==1 else '−1',18)
p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#63d7ed" stroke-width="3"/>');text(p,28,425,'Frame 5: score −1 → suppress output of a confirmed track, retain state.',18);text(p,28,461,'Frame 6: evidence recovers. Toy arithmetic, not the 15-frame confirmation policy.',17);finish('timeline',p)
rows=[('OWLv2',17.3),('OWLv2*',24.6),('LLMDet-L',6.5),('DINO-X',21.3),('SAM 3',54.1),('Human oracle',72.8)]
p=start('Measured image PCS · SA-Co/Gold · Table 1',670);text(p,28,40,'Measured image PCS · SA-Co/Gold · Table 1',24);text(p,28,80,'Instance cgF1 ↑ · text prompt · zero-origin bars',18)
for i,(name,value) in enumerate(rows):
 y=115+i*72;text(p,28,y+27,name,18);p.append(f'<rect x="235" y="{y}" width="{value*7}" height="36" fill="{GREEN if i==5 else BLUE}"/>');text(p,770,y+27,str(value),19)
p.append('<path d="M235 557 H795" stroke="#8299b8"/>')
for tick in range(0,81,20):text(p,228+tick*7,588,str(tick),16)
text(p,28,622,'* OWLv2 partially trained on LVIS; DINO-X from API; no uncertainty bars.',17);text(p,28,652,'Gold selects best of 3 annotations; human is an oracle agreement reference.',17);finish('results',p)
assert all(isclose(a,b) for a,b in zip([.8*.9,.8*.7,.8*.4],[.72,.56,.32]))
assert isclose(-20*log(.8),4.462871026284194)
assert scores==[1,2,1,0,-1,0,1]
assert isclose(100*.75*((8*8-2*2)/sqrt(10**4)),45)
print('Eight diagrams generated; score, BCE, MDS and metric arithmetic passed.')
