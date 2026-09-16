"""Reproduce Qwen-Image teaching SVGs; measured values from course PDF Table 9."""
from pathlib import Path
from html import escape
out=Path(__file__).resolve().parents[1]/'public/figures/qwen-image'
out.mkdir(parents=True,exist_ok=True)
def text(x,y,s,size=19):
    return f'<text x="{x}" y="{y}" font-size="{size}">{escape(s)}</text>'
def box(x,y,w,a,b,color='#ddd9fa'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="86" rx="12" fill="{color}" stroke="#71849a"/>'+text(x+14,y+32,a)+text(x+14,y+62,b,16)
def arrow(x,y,a,b):
    return f'<path d="M{x} {y} L{a} {b}" stroke="#53677d" stroke-width="3" fill="none" marker-end="url(#a)"/>'
def svg(name,title,desc,body,h=440):
    (out/f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 {h}" role="img" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#53677d"/></marker></defs><rect width="940" height="{h}" rx="18" fill="#f3f6fb"/><g font-family="Arial,sans-serif" fill="#17304b">'+text(28,43,title,25)+body+'</g></svg>')
svg('problem','01 / Three tests for the same poster','Schematic goals: symbols, instruction, and preservation.',box(30,115,260,'Spell the words','OPEN AT NINE','#d4eee7')+box(340,115,260,'Follow the edit','Change the lettering')+box(650,115,260,'Preserve the scene','Keep cup and lighting','#ffe6cb')+text(30,285,'Beautiful pixels can contain wrong letters. Correct edits can lose details.',19)+text(30,335,'Data coverage → latent generation → faithful decoding all matter.',19)+text(30,401,'Teaching scenario. No image model outputs or benchmark measurements.',18))
svg('architecture','02 / Generate an image latent under language guidance','Functional T2I architecture; the target state is repeatedly updated.',box(30,95,260,'User prompt S','Poster with exact text')+arrow(300,138,335,138)+box(345,95,250,'A · Qwen2.5-VL','Last hidden states h')+arrow(470,191,470,245)+box(30,255,260,'B · State xₜ','Patchify latent + time t','#d4eee7')+arrow(300,298,335,298)+box(345,255,250,'C · MMDiT','60 blocks · predict velocity')+arrow(605,298,640,298)+box(650,255,260,'Update image state','Carry forward to next t','#d4eee7')+arrow(780,351,780,405)+box(650,415,260,'D · Image decoder','Final latent → pixels','#ffe6cb')+text(30,450,'Repeat C + update until data endpoint.',19)+text(30,490,'20B velocity model; 7B condition encoder.',18)+text(30,559,'Schematic: projections and block internals omitted. Decode after the flow.',18),590)
svg('representation','03 / Compression and reconstruction are different jobs','Illustrative 256-square input and dimensions calculated from Table 1.',box(30,110,260,'Image · 256 × 256','VAE encoder E: ÷8','#d4eee7')+arrow(300,153,335,153)+box(345,110,250,'Latent · 32 × 32 × 16','2 × 2 latent patches','#d4eee7')+arrow(605,153,640,153)+box(650,110,260,'256 image tokens','64 values before embedding')+box(30,280,260,'Freeze VAE encoder','Shared image/video code','#d4eee7')+arrow(300,323,335,323)+box(345,280,250,'Adapt image decoder','Train on text-rich images','#ffe6cb')+arrow(605,323,640,323)+box(650,280,260,'Reconstruct fine detail','Not a spelling guarantee','#ffe6cb')+text(30,435,'Top: tensor arithmetic. Bottom: VAE adaptation stage, not all training.',18),465)
svg('training','04 / Known target during training; unknown during sampling','Flow loss trains weights; sampling only changes the image latent.',box(30,110,260,'TRAIN · Target I → z','Noise ε; sample time t','#d4eee7')+arrow(300,153,335,153)+box(345,110,250,'Mix · xₜ = tz + (1−t)ε','C predicts vθ(xₜ, t, h)')+arrow(605,153,640,153)+box(650,110,260,'Compare with z − ε','Squared error → update θ','#ffe6cb')+box(30,285,260,'SAMPLE · Start noise','t = 0; fixed conditioning','#d4eee7')+arrow(300,328,335,328)+box(345,285,250,'C · Predict and update','Advance t toward 1')+arrow(605,328,640,328)+box(650,285,260,'D · Decode final latent','Fixed model weights','#ffe6cb')+text(30,431,'A supplies h on both paths. The clean target z is absent at inference.',18),465)
svg('numbers','05 / One coordinate through the flow boxes','Synthetic scalar trace, not model inference.',box(30,110,260,'B · z = 2; ε = 10','t = 0.25 gives xₜ = 8','#d4eee7')+arrow(300,153,335,153)+box(345,110,250,'C · Predict −7','Target z − ε = −8')+arrow(605,153,640,153)+box(650,110,260,'Training error','(−7 − (−8))² = 1','#ffe6cb')+box(30,285,260,'Sampling · xₜ = 8','Choose Δt = 0.25','#d4eee7')+arrow(300,328,335,328)+box(345,285,250,'Add Δt × prediction','8 + 0.25 × (−7)')+arrow(605,328,640,328)+box(650,285,260,'New state = 6.25','Oracle would give 6','#d4eee7')+text(30,431,'A supplies fixed h. B is assumed encoded. D is not a scalar identity.',18),465)
body=text(30,98,'Oracle velocity −8; uniform illustrative steps Δt = 0.25',20)
for i,(t,v) in enumerate([('0','10'),('0.25','8'),('0.5','6'),('0.75','4'),('1','2')]):
 x=30+i*180
 body+=box(x,150,150,'t = '+t,'state = '+v,'#d4eee7')
 if i<4: body+=arrow(x+154,193,x+175,193)
body+=text(30,295,'NOISE',18)+text(750,295,'DATA',18)+text(30,352,'Carry image state forward. Keep prompt features h and weights fixed.',19)+text(30,404,'Teaching oracle path, not the reported sampling schedule or a real model.',18)
svg('steps','06 / Qwen flow time runs from noise to data', 'Five scalar states along a four-step oracle trajectory.',body)
body=''
for row in range(3):
 for col in range(3):
  x,y=45+col*100,105+row*80
  body+=f'<rect x="{x}" y="{y}" width="90" height="70" rx="8" fill="#d4eee7"/>'+text(x+15,y+40,f'({col-1},{row-1})',18)
for i,(label,coord) in enumerate([('a','(2, 2)'),('cute','(3, 3)'),('cat','(4, 4)')]):
 x,y=400+i*160,105+i*75
 body+=box(x,y,145,label,coord)
body+=text(45,382,'Image: centered 2D grid',19)+text(400,382,'Text: shared ID across both axes',19)+text(30,439,'Editing adds frame ID to distinguish reference and target image tokens.',18)+text(30,480,'Position IDs are not word placement instructions or flow time t.',18)
svg('positions','07 / MSRoPE preserves two geometries','Schematic centered image grid and diagonal text positions adapted from Figure 8.',body,515)
svg('editing','08 / Two paths from one reference','Semantic features and clean reconstructive tokens condition the evolving target.',box(30,95,260,'Reference + instruction','Understand requested change')+arrow(300,138,335,138)+box(345,95,250,'A · Qwen2.5-VL','Semantic conditioning h')+arrow(470,191,470,285)+box(30,285,260,'Reference → VAE','Clean reference tokens','#ffe6cb')+arrow(300,328,335,328)+box(345,295,250,'C · MMDiT','Joint attention')+arrow(605,338,640,338)+box(650,295,260,'Update target only','Then D: decode at t = 1','#d4eee7')+box(345,470,250,'Noisy target tokens','Sequence-concatenate','#d4eee7')+arrow(470,460,470,391)+text(30,590,'Reference features stay fixed. Frame IDs separate reference and target.',18)+text(30,630,'Schematic paths; fixed conditioning is not a pixel-preservation constraint.',18),660)
svg('curriculum','09 / Teach coverage, context, and layout','Two complementary progressions; columns are not synchronized training stages.',box(30,105,260,'256 × 256','Learn broad visual structure','#d4eee7')+arrow(300,148,335,148)+box(345,105,250,'640 × 640','Increase detail','#d4eee7')+arrow(605,148,640,148)+box(650,105,260,'1328 × 1328','Refine high-resolution data','#d4eee7')+box(30,285,260,'Pure rendering','Text on clean backgrounds')+box(345,285,250,'Text in context','Compositional rendering')+box(650,285,260,'Complex rendering','Structured templates')+text(30,433,'Schematic: multiple aspect ratios, stricter filtering, and balanced sampling.',18)+text(30,476,'The report does not equate the three synthesis types with these resolutions.',18),510)
body=text(30,94,'ChineseWord · single-character rendering · Table 9 · accuracy ↑',19)
for i,(label,val) in enumerate([('Level 1',97.29),('Level 2',40.53),('Level 3',6.48)]):
 y=125+i*80
 body+=text(30,y+27,label)+f'<rect x="200" y="{y}" width="{val*6}" height="38" rx="5" fill="#168373"/>'+text(210+val*6,y+27,f'{val:.2f}%')
for v in [0,25,50,75,100]: body+=text(200+v*6,385,str(v),16)
body+=text(30,438,'Measured v1 results. Rare-character failures remain despite the overall lead.',18)
svg('results','10 / Common characters are much easier than rare ones','Qwen-Image: Level 1 97.29%, Level 2 40.53%, Level 3 6.48%.',body,470)
