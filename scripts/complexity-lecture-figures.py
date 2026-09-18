"""Reproduce lecture-guide schematics and check the displayed toy arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/complexity-lecture')
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

# All schematic scores below are teaching inputs, not CTM table entries.
diagram('codes','Describe structure; count the whole code',[
('SAME LENGTH · different candidate descriptions', [('100111010110','Literal: twelve payload bits'),('100100100100','Motif 100, repeated four times'),('Decoder + syntax','Overhead matters at this size')],False),
('TWO-PART TOY · shared inputs; model plus labels', [('Model A','20 + 80 = 100 bits'),('Model B','60 + 20 = 80 bits'),('Compare total descriptions','Not just model size')],False)])
diagram('kl','A continuous point does not have finite prior mass',[
('GAUSSIAN TOY · P = N(0,1); Q = N(0,σ²)', [('σ = 1','KL = 0 bits'),('σ = 0.5','KL ≈ 0.459 bits'),('σ = 0.1','KL ≈ 2.608 bits')],False),
('VARIATIONAL OBJECTIVE · likelihood loss and KL in bits', [('Expected data code','E_Q[−log₂ p(S | w)]'),('Prior-relative cost','KL₂(Q || P)'),('Sum ≥ −log₂ p(S)','Gap: KL to exact posterior')],False),
('POINT LIMIT · σ → 0', [('Variance shrinks','Precision cost grows'),('KL diverges','Density is not point mass'),('Discrete point differs','Cost = −log₂ P(point)')],False)])
diagram('bdm','Block repetition is cheap; global order is missing',[
('ILLUSTRATIVE LOOKUPS · not empirical CTM values', [('A, A, A, B','Partitioned object'),('A costs 5; count 3','B costs 7; count 1'),('Sum distinct-block terms','5 + log₂ 3 + 7 = 13.585')],True),
('REORDER THE SAME BLOCKS', [('A, B, A, A','Same block multiset'),('Same lookup costs','Same multiplicities'),('Same BDM score','Different ordering invisible')],True)])
diagram('pipeline','QuBD reads weights; it does not update this predictor',[
('TRAIN · repeated supervised updates', [('x ∈ ℝᵃ → f_w(x) ∈ ℝᴷ','Inputs → prediction'),('Loss against target y','Backpropagate derivatives'),('Optimizer updates w ∈ ℝᵈ','Next batch repeats the loop')],True),
('DIAGNOSE · read a checkpoint w', [('Uniform quantizer Q_b','d real weights → d integers'),('Split into b binary planes','Each plane has d entries'),('BDM per plane; sum','Normalize vs. random reference')],True),
('INFER · fixed predictor', [('Fresh x*','No target required'),('Frozen weights w','Run f_w(x*)'),('Prediction','Diagnostic need not run')],True)])
diagram('planes','Four weights through a two-bit representation',[
('TOY QUANTIZER · range [−1,1]; b = 2; no clipping', [('w = (−1, −1/3, 1/3, 1)','q = round(3(w + 1)/2)'),('q = (0, 1, 2, 3)','Binary: 00, 01, 10, 11'),('Reconstruct real weights','ŵ = −1 + 2q/3')],True),
('PLANES · read corresponding entries down the two vectors', [('High: β₁ = (0,0,1,1)','Low: β₀ = (0,1,0,1)'),('Integer reconstruction','q = 2β₁ + β₀'),('Drop the low plane','Lose residual (0,1,0,1)')],True),
('ASSUMED SCORES · only arithmetic, no CTM evaluation', [('BDM(β₁) = 6','BDM(β₀) = 8'),('QuBD = 6 + 8 = 14','Random reference = 20'),('Normalized = 14/20','0.7 = 70%; not a probability')],True)])
diagram('momos','MoMos: update, constrain, carry forward',[
('EACH ITERATION · start from previous constrained weights', [('Optimizer step','Temporary dense w⁽ᵗ⁾'),('Partition ψ','m blocks, s values each'),('Build k motifs','Zero + k−1 sampled blocks')],True),
('PROJECTION · no fixed codebook assumed across iterations', [('Nearest-motif assignment','Minimize squared distance'),('Reconstruct with ψ⁻¹','Replace blocks by motifs'),('Carry constrained ŵ⁽ᵗ⁾','Input to next optimizer step')],True),
('INFERENCE · freeze final motifs and mosaic', [('Dictionary Z; indices M','No resampling or optimizer'),('Reconstruct final weights','Same predictor architecture'),('Evaluate fresh x*','Speedup needs implementation')],True)])
diagram('motif-example','One projection; shared motifs replace near matches',[
('SAMPLED TOY · m = 3; s = 2; c = 0.8; k = 2', [('Block 1: (0.1,0.2)','Distances²: 0.05, 1.45'),('Block 2: (0.9,1.1)','Distances²: 2.02, 0.02'),('Block 3: (1,1)','Distances²: 2, 0')],False),
('MOTIFS · z₁ = (0,0); z₂ = sampled block 3 = (1,1)', [('M(1) = 1','Replace by (0,0)'),('M(2) = 2','Replace by (1,1)'),('M(3) = 2','Keep (1,1)')],False),
('SEPARATE STORAGE TOY · m = 16; s = 4; k = 2; 32 bits/scalar', [('Dense: 16 × 4 × 32','2048 bits'),('Dictionary: 2 × 4 × 32','256 bits'),('Indices: 16 × 1','Total 272, before overhead')],False)])
assert math.isclose((.25-1-math.log(.25))/(2*math.log(2)), .458989359667, abs_tol=1e-9)
assert math.isclose(5+math.log2(3)+7,13.58496250,abs_tol=1e-8)
assert 2*4*32+16==272
# Preserve an original source slide; Poppler is needed only to regenerate it.
import shutil, subprocess
pdf=Path('public/papers/complexity-lecture.pdf')
poppler=shutil.which('pdftoppm')
if poppler:
    subprocess.run([poppler,'-f','32','-l','32','-scale-to','1800','-png','-singlefile',str(pdf),str(out/'source-results')],check=True)
else:
    print('pdftoppm unavailable: retained existing source-results.png; install Poppler to regenerate.')
print('Seven schematic SVGs generated; toy arithmetic checked.')
