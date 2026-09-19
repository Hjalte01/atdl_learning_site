"""Reproduce Mosaic-of-Motifs schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/momos-lecture')
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

diagram('reuse','Same weight slots; different descriptions',[
('DENSE · sixteen blocks, four scalars each', [('Store every block','16 × 4 = 64 scalar values'),('Original arrangement','Each location has its values'),('Run the predictor','64 slots remain in the model')],True),
('MOTIFS · three distinct patterns in sixteen positions', [('Store A, B, C','3 × 4 = 12 scalar values'),('Store the mosaic','16 indices specify locations'),('Reconstruct the same array','When all blocks match exactly')],True)])
diagram('representation','Bank Z + mosaic M → reconstructed weights',[
('BANK · each letter names a four-scalar block', [('A = (0,0,0,0)','One stored motif'),('B = (1,1,1,1)','One stored motif'),('C = (−1,0,0,−1)','One stored motif')],False),
('MOSAIC · sixteen positions; counts A:7, B:5, C:4', [('Row 1: A B A C','Row 2: C A B A'),('Row 3: B C A B','Row 4: A B C A'),('Each position is an index','Counts alone lose location')],False),
('LOOKUP · first row reconstructed, with the fixed layout', [('A B','(0,0,0,0), (1,1,1,1)'),('A C','(0,0,0,0), (−1,0,0,−1)'),('Repeat for other rows','ψ⁻¹ restores tensor positions')],False)])
diagram('storage','A complete payload includes the mosaic',[
('TEACHING BUDGET · n = 64; s = 4; m = 16; k = 3; q = 32', [('Dense array','64 × 32 = 2048 bits'),('Motif bank','3 × 4 × 32 = 384 bits'),('Index for every block','16 × ceil(log₂ 3) = 32 bits')],False),
('COMPARE · no headers, alignment or decoder costs included', [('Bank alone: 384 bits','Missing locations'),('Bank + mosaic: 416 bits','2048 / 416 ≈ 4.92'),('Not a measured file ratio','Not an inference speedup')],True)])
diagram('training','Propose → constrain → carry forward',[
('TRAIN · previous constrained weights Ŵ; input x; target y', [('Predict f_Ŵ(x)','Evaluate task loss ℓ'),('Optimizer step','Temporary proposal U ∈ ℝⁿ'),('Partition ψ(U)','m blocks bᵢ ∈ ℝˢ')],True),
('CONSTRAIN · continue these operations each iteration', [('Build bank Z ∈ ℝᵏˣˢ','Zero + sampled blocks'),('Nearest motif for each bᵢ','M ∈ {1,…,k}ᵐ'),('Reconstruct Ŵ = ψ⁻¹(Z[M])','Carry Ŵ to next train step')],True),
('INFER · freeze final Z and M; no resampling', [('New input x*','No training target'),('Reconstruct final Ŵ','Evaluate f_Ŵ(x*)'),('Output prediction','Dense compute need not shrink')],True)])
diagram('distances','Nearest motif: one projection with k = 2',[
('BANK · motif 1 = (0,0); motif 2 = (1,1); s = 2', [('Block 1: (0.1,0.2)','Squared errors: 0.05, 1.45'),('Block 2: (0.9,1.1)','Squared errors: 2.02, 0.02'),('Block 3: (1,1)','Squared errors: 2, 0')],False),
('ASSIGN THE MINIMUM · each position makes its own choice', [('Index 1 → (0,0)','Projection error² = 0.05'),('Index 2 → (1,1)','Projection error² = 0.02'),('Index 2 → (1,1)','Projection error² = 0')],False),
('RECONSTRUCT · M = (1,2,2)', [('Ŵ = (0,0,1,1,1,1)','Total squared error = 0.07'),('Small weight distortion','Does not bound task loss alone'),('Illustrative values','Not trained-model evidence')],False)])
diagram('iterations','The constrained state enters the next step',[
('STEP t · η = 0.1; gradient stipulated for this example', [('Start: (0,0,1,1,1,1)','g = (−1,−2,1,−1,0,0)'),('Proposal blocks','(.1,.2), (.9,1.1), (1,1)'),('Bank: (0,0), (1,1)','Carry: (0,0,1,1,1,1)')],True),
('STEP t+1 · start from that carried state, not the old proposal', [('g = (0,0,−2,−2,−2,−2)','Same learning rate'),('Proposal blocks','(0,0), (1.2,1.2), (1.2,1.2)'),('Bank: (0,0), (1.2,1.2)','Carry proposal exactly')],True),
('WHAT CHANGED?', [('Same mosaic: (1,2,2)','Motif values changed'),('Rebuild inside the loop','Bank is not fixed forever'),('No convergence claim','Toy gradients are assumed')],False)])
diagram('domain','Smaller pieces allow finer recombination',[
('FIXED n = 16 AND k = 2 · distinct fixed motifs; toy count', [('s = 2; m = 8 positions','2⁸ = 256 index strings'),('Bank stores 4 scalars','Capacity k/m = 1/4'),('8 one-bit indices','Each block can choose A or B')],False),
('CHANGE BLOCK SIZE · keep n and k fixed', [('s = 4; m = 4 positions','2⁴ = 16 index strings'),('Bank stores 8 scalars','Capacity k/m = 1/2'),('4 one-bit indices','Larger coupled weight groups')],False),
('COMPARISON LIMIT', [('Capacity is not held fixed','Neither is total storage'),('Duplicate motifs collapse','different strings to same array'),('Measure task performance','Counting is not an accuracy law')],False)])
assert 3*4*32+16*math.ceil(math.log2(3))==416
assert math.isclose(2048/416,4.923076923076923)
blocks=[(.1,.2),(.9,1.1),(1,1)]
bank=[(0,0),(1,1)]
dist=[[sum((a-b)**2 for a,b in zip(x,z)) for z in bank] for x in blocks]
assert [min(range(2),key=lambda j:r[j]) for r in dist]==[0,1,1]
assert math.isclose(sum(min(r) for r in dist),.07)
assert math.isclose(-.9*math.log2(.9)-.1*math.log2(.1),.4689955935892812)
import shutil, subprocess
poppler=shutil.which('pdftoppm')
if poppler:
    subprocess.run([poppler,'-f','36','-l','36','-scale-to','1800','-png','-singlefile','public/papers/momos-lecture.pdf',str(out/'source-results')],check=True)
else:
    print('Retained source-results.png; use Poppler pdftoppm to regenerate PDF page 36.')
print('Seven teaching diagrams generated; storage, assignment and entropy examples checked.')
