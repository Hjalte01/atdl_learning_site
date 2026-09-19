"""Reproduce Chapter 10 neural compression schematics and check teaching arithmetic."""
from pathlib import Path
from html import escape
import math
out = Path('public/figures/neural-compression')
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


diagram('codec','A latent vector becomes a message only after discrete coding',[
('ENCODE · x ∈ ℝᴰ → y ∈ ℝᴹ → indices i ∈ {0,…,K−1}ᴹ', [('A: neural encoder','x → continuous y'),('B: quantizer + codebook','y → indices i; values c[i]'),('C: entropy encoder','i + probabilities → bits')],True),
('DECODE · shared codebook, probability model and conventions', [('D: entropy decoder','bits + probabilities → i'),('B: codebook lookup','i → quantized values c[i]'),('E: neural decoder','c[i] → reconstruction x̂')],True),
('LOSS LOCATION · lossy transform/quantization versus exact symbol recovery', [('x may differ from x̂','Lossy end-to-end codec'),('i must match exactly','Lossless entropy stage'),('F: probability model pλ','Supplies C and D')],False)])
diagram('paths','Three paths through the same learned components',[
('TRAIN · update encoder, decoder, codebook and probability model', [('x → A → soft B → E','Compute distortion'),('F predicts code probabilities','Compute rate surrogate'),('Distortion + β × rate','Backpropagate; no bitstream')],True),
('COMPRESS · freeze all learned parameters', [('x → A → hard B','Obtain integer symbols'),('F + C → transmitted bits','D + F → same symbols'),('Lookup codebook → E','Reconstruct this image')],True),
('GENERATE · freeze parameters; no transmitted image', [('F: sample first index','Draw from its distribution'),('F: sample next index','Condition on sampled prefix'),('Lookup codebook → E','Synthesize a new image')],True)])
diagram('quantize','Close-up: floats, indices and codebook values are different objects',[
('HARD ASSIGNMENT · codebook c = (−1,0,1); nearest value', [('y = (−0.8,0.2,0.9)','Three continuous values'),('i = (0,1,2)','Three integer indices'),('c[i] = (−1,0,1)','Three reconstructed values')],True),
('SOFT TRAINING · one row per coordinate, one column per codebook entry', [('Similarity matrix S','Shape M × K'),('W = softmax(τ S)','Rows sum to 1'),('Soft values = Wc','Generally between entries')],True),
('LIMITS · raising τ sharpens assignments but can saturate gradients', [('Small τ','Diffuse mixtures'),('Large τ; unique winner','Approaches one-hot'),('Exact tie','Mixture can persist')],False)])
diagram('worked','Trace a complete toy codec: x = (0.2,0.8)',[
('A → B · identity encoder; shared codebook c = (0,1)', [('A: y = (0.2,0.8)','Two real numbers'),('B: i = (0,1)','Nearest-entry quantization'),('F: p(0)=¾, p(1|0)=½','Sequence mass = ⅜')],True),
('C → D · illustrative prefix codes, known sequence length = 2', [('C: 0 → bit 0; 1 → bit 1','Transmit 01: exactly 2 bits'),('D: read 0, then 1','Recover i = (0,1)'),('Lookup c[i] = (0,1)','E: identity → x̂ = (0,1)')],True),
('MEASURE · distortion is separate from ideal probability-based rate', [('MSE = (0.04+0.04)/2','= 0.04'),('Ideal rate = −log₂(⅜)','≈ 1.415 bits per image'),('Actual toy payload = 2 bits','Headers excluded')],False)])
diagram('sequential','Autoregressive decoding carries an exact recovered prefix',[
('STEP 1 · empty prefix available to both ends', [('F predicts p(i₁)','Same distribution at C and D'),('C encodes known i₁','D decodes i₁ from bits'),('Carry prefix (i₁)','Do not sample it')],True),
('STEP 2 · only earlier symbols may affect current probabilities', [('F predicts p(i₂ | i₁)','No future-symbol leakage'),('C encodes known i₂','D recovers the same i₂'),('Carry prefix (i₁,i₂)','Repeat until known length')],True),
('GENERATION USES THE SAME F BUT A DIFFERENT OPERATION', [('Predict next probabilities','Use generated prefix'),('Sample next symbol','No compressed message'),('Append sampled symbol','Different task from decoding')],True)])
diagram('rate','Bit accounting: symbol count is not a probability',[
('FIXED WIDTH · M = 4 coordinates, K = 4 entries, 16 grayscale pixels', [('log₂ K = 2 bits / index','4 indices × 2 bits'),('Payload = 8 bits','Sequence probability = 4⁻⁴'),('8 / 16 = 0.5 bpp','Shared model excluded')],True),
('MODEL-BASED IDEAL RATE · illustrative conditional probabilities', [('Probabilities: ½, ½, ¼, ½','Sequence mass = 1/32'),('−log₂(1/32) = 5 bits','5 / 16 = 0.3125 bpp'),('Measure real stream too','Include headers / side info')],True)])
diagram('tradeoff','A rate penalty changes which reconstruction is preferred',[
('TWO CONSTRUCTED CANDIDATES · use the same distortion and rate units', [('Candidate A','d = 0.01; r = 4 bits'),('Candidate B','d = 0.04; r = 1 bit'),('Objective: d + βr','Compare, do not average')],False),
('β = 0.005 · cheaper distortion matters more', [('A: 0.01 + 0.005 × 4','= 0.030'),('B: 0.04 + 0.005 × 1','= 0.045'),('Prefer candidate A','More bits; less distortion')],False),
('β = 0.02 · rate is more expensive', [('A: 0.01 + 0.02 × 4','= 0.090'),('B: 0.04 + 0.02 × 1','= 0.060'),('Prefer candidate B','Fewer bits; more distortion')],False)])
assert math.isclose(-math.log2(.75*.5),1.415037499278844)
assert math.isclose((.2**2+(.8-1)**2)/2,.04)
assert 4*math.log2(4)/16==.5
assert -math.log2(.5*.5*.25*.5)/16==.3125
# Optional source-page reproduction: POPPLER_PDFTOPPM=/path/to/pdftoppm python3 scripts/neural-compression-figures.py
import os,subprocess
if os.environ.get('POPPLER_PDFTOPPM'):
    subprocess.run([os.environ['POPPLER_PDFTOPPM'],'-f','287','-l','287','-scale-to','1500','-png','-singlefile','public/materials/52f0790c3323bf24.pdf',str(out/'source-figure-10-5')],check=True)
