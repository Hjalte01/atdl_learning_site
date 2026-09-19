---
title: "Deep Generative Modeling · from latent codes to compressed bits"
shortTitle: "Neural compression"
topic: "topic-2"
order: 3
materialKind: "extra"
description: "Follow Chapter 10 from learned transforms through quantization, probability models and real bitstreams, with worked rate–distortion examples."
status: "ready"
pdf: "/papers/deep-generative-modeling-compression.pdf"
---
## The puzzle: is a small latent vector already a compressed file?

Your autoencoder maps an image to a handful of floats and reconstructs it beautifully. Can you send those floats and declare victory over an image codec? You still need to specify their precision, encode them into bits, and tell the receiver how to recover them. **A useful representation becomes a codec only when the receiver can decode an actual message.**

This guide follows **Chapter 10, printed pp. 259–273**, of Jakub M. Tomczak’s *Deep Generative Modeling*, second edition, Springer, 2024, ISBN 978-3-031-64087-2. [Open this exact course copy](/papers/deep-generative-modeling-compression.pdf). Its second-edition preface is dated May 2, 2024. The catalog contains two different PDF files of this edition; this guide uses `topic 2/Deep Generative Modeling.pdf` (325 PDF pages). It complements the [existing model-family roadmap](/papers/t2-deep-generative-modeling), which uses the other copy. Different file hashes do not establish different scientific editions.

Allow 25 minutes. You need conditional probability, logarithms and basic gradients. This is a focused chapter guide, not an exhaustive second summary of the book. All seven SVGs and numerical examples are teaching constructions; the final source figure is explicitly identified. Open figures to enlarge them. No trained codec runs here.

## 1 · Architecture: where does information disappear?

[![Encoder, quantizer and entropy encoder send bits to an entropy decoder, codebook lookup and neural decoder; a shared probability model supports both entropy stages.](/figures/neural-compression/codec.svg)](/figures/neural-compression/codec.svg)

Read the first row at the sender and the second at the receiver. **A** transforms an image $x\in\mathbb R^D$ into $y\in\mathbb R^M$. **B** maps each coordinate to an index $i_m$ in a shared codebook $c\in\mathbb R^K$, giving quantized values $\hat y_m=c_{i_m}$. **C** turns those indices into bits using probabilities supplied by **F**. **D** reverses that entropy-coding operation, then codebook lookup and **E** produce $\hat x$.

Entropy decoding must recover the **indices exactly**. The final image may still differ from the input: the transform and quantizer can discard information. A lossless entropy stage inside a lossy codec is entirely consistent. An invertible continuous transform alone also does not guarantee lossless recovery after finite-precision quantization.

This functional diagram follows §§10.2 and 10.4, Figures 10.1 and 10.4. The book gives illustrative fully connected encoder/decoder networks in Listing 10.2; their widths are parameters, not a universal codec architecture. Sender and receiver must agree on the probability model, codebook, symbol order, numerical coding conventions and message length or termination rule. These are requirements for a decodable system, not new network layers.

The design reasoning is our interpretation: learn a transform that makes useful information cheap to describe, make its outputs discrete, then exploit their predictable structure. The book motivates replacing fixed transforms with learned ones; it does not establish that every learned transform beats JPEG.

<details><summary>Predict: if the entropy decoder is perfect, must the image reconstruction be perfect?</summary>

No. It recovers the quantized indices. Information removed before entropy coding is not restored by lossless symbol recovery.

</details>

## 2 · Training, compression and generation follow different paths

[![Three rows distinguish differentiable training without bit coding, frozen compression through a message, and unconditional generation by sampling indices.](/figures/neural-compression/paths.svg)](/figures/neural-compression/paths.svg)

During **training**, compute reconstruction and a probability-based rate surrogate, then update encoder weights $\phi$, decoder weights $\theta$, entropy-model weights $\lambda$ and the codebook $c$. The actual entropy coder need not run in the gradient path (§10.4.4).

During **compression**, freeze those learned quantities, encode a particular image, make hard symbol decisions and transmit a bitstream. During **generation**, start with no image: sample a sequence from F, look up its codebook values and run E. The latter produces a new image; it does not decode somebody’s compressed photograph.

Source Eq. 10.9 becomes the following explicitly normalized teaching objective:

$$
\mathcal L(\phi,\theta,\lambda,c)=\frac1N\sum_{n=1}^N\left[d(x_n,\hat x_n)+\beta r(i_n)\right],\qquad
\hat x_n=f_{d,\theta}(c[i_n]),\quad i_n=Q_{\rm hard}(f_{e,\phi}(x_n);c).
$$

Here $N$ is the number of examples, $d$ is distortion (we use mean squared error), $r$ is a code-length estimate, and $\beta>0$ makes bits costly relative to distortion. **The distortion term spans A→B→E; rate connects B to F.** This hard-assignment expression describes the intended objective; training needs a gradient surrogate at B, explained next. It is not a claim that exact argmin is differentiable.

The source uses natural logarithms, so its rate is in **nats**. Our examples use $r=-\log_2p_\lambda(i)$ in **bits**. Since $-\ln p=(\ln2)(-\log_2p)$, the coefficient must change when comparing those conventions. Listing 10.5 also averages rate over latent coordinates, so its numeric $\beta$ is not directly interchangeable with a total-bits objective.

## 3 · Representation close-up: an index is not a float

[![Continuous values minus 0.8, 0.2 and 0.9 map to indices 0, 1 and 2 in codebook minus 1, 0, 1; soft training instead forms weighted codebook mixtures.](/figures/neural-compression/quantize.svg)](/figures/neural-compression/quantize.svg)

For hard scalar quantization, coordinate $m$ chooses the nearest of $K$ entries:

$$
i_m=\arg\min_{k\in\{0,\ldots,K-1\}}(y_m-c_k)^2,\qquad \hat y_m=c_{i_m}.
$$

This is box B. With $c=(-1,0,1)$ and $y=(-0.8,0.2,0.9)$, the indices are $(0,1,2)$ and values are $(-1,0,1)$. The message represents the indices; storing three arbitrary floats is unnecessary when the codebook is already shared. A tie-breaking rule is needed for exactly equidistant entries.

The chapter’s prose constructs similarity scores and a soft assignment matrix:

$$
S_{mk}=\exp\bigl(-(y_m-c_k)^2\bigr),\qquad
W_{mk}=\frac{\exp(\tau S_{mk})}{\sum_{j=0}^{K-1}\exp(\tau S_{mj})},\qquad
\tilde y=Wc.
$$

$S,W\in\mathbb R^{M\times K}$ have one row per coordinate; $\tau>0$ sharpens each row, and $\tilde y\in\mathbb R^M$ enters E during soft training. The final matrix multiplication is the mechanism in source Eq. 10.3. A weighted mixture provides gradients to the codebook and encoder.

**Soft is not exactly discrete.** At finite $\tau$, entries generally remain mixtures; large $\tau$ approaches one-hot assignments only with a unique winner. Exact ties can remain mixed. Numerically saturated softmax can also give nearly zero gradients despite formal differentiability. For $y=0.5$ and $c=(0,1)$, the two similarities tie: soft output stays $0.5$, while hard quantization chooses one endpoint. This exposes a real training/deployment mismatch.

There is a source implementation distinction: Listing 10.3 computes $\exp(-|y_m-c_k|)$, while the prose uses squared distance inside the exponential. Both rank nearest entries identically but yield different soft weights and gradients. We follow the prose formula above and do not present the listing as a verified drop-in codec.

## 4 · Worked example: follow one image all the way back

[![A two-pixel image 0.2, 0.8 passes through identity transforms and binary quantization, transmits bits 01, and reconstructs 0, 1 with MSE 0.04.](/figures/neural-compression/worked.svg)](/figures/neural-compression/worked.svg)

Use a deliberately tiny grayscale “image” $x=(0.2,0.8)$, identity A and E, and $c=(0,1)$. B chooses $i=(0,1)$, so E reconstructs $\hat x=(0,1)$. Distortion is

$$
d(x,\hat x)=\frac{(0.2-0)^2+(0.8-1)^2}{2}=0.04.
$$

Let F assign $p(i_1=0)=3/4$ and $p(i_2=1\mid i_1=0)=1/2$. The sequence probability is $3/8$, and its ideal information content is $-\log_2(3/8)\approx1.415$ bits.

For an actual, easily checked message, use the binary prefix code $0\mapsto0$, $1\mapsto1$ at each position, with length two known in advance. C emits **01**; D reads the two bits and recovers $(0,1)$. The payload is **2 bits**, or 1 bit per pixel, before headers. This simple prefix code does not achieve the fractional ideal length. Arithmetic coding can exploit sequence probabilities more closely over suitable messages; a single tiny message still has termination and precision costs.

Thus the example follows every architecture box while keeping three different measurements visible: reconstruction error 0.04, ideal rate 1.415 bits, actual toy payload 2 bits. No trained network or real compression benchmark is implied.

## 5 · Probability becomes useful through synchronized decoding

[![Each autoregressive decoding step predicts a distribution from the already decoded prefix, recovers the next symbol from bits and carries that prefix onward; generation samples instead.](/figures/neural-compression/sequential.svg)](/figures/neural-compression/sequential.svg)

The autoregressive entropy model in §10.4.3 represents

$$
p_\lambda(i)=\prod_{m=1}^{M}p_\lambda(i_m\mid i_{<m}),\qquad
r(i)=-\sum_{m=1}^{M}\log_2p_\lambda(i_m\mid i_{<m}).
$$

$i_{<m}$ is the prefix of earlier symbols. F supplies these conditional probabilities to both C and D. C already knows the current symbol; D recovers it from the bitstream, then both advance to the same prefix. D **does not randomly sample** its next symbol. If probabilities depend on future values unavailable at D, the procedure is not decodable as described. This explains why causal conditioning matters beyond ordinary predictive accuracy.

Training can evaluate all observed target positions under a causal model; decompression still needs the recovered prefix step by step. The source’s `ARMEntropyCoding` class is a probability model, explicitly not a complete arithmetic-coder implementation.

A useful teaching identity explains the rate term. For the actual distribution $q$ over discrete index sequences,

$$
\mathbb E_{i\sim q}[-\log_2p_\lambda(i)]=H_2(q)+D_{\rm KL,2}(q\|p_\lambda).
$$

$H_2$ is entropy in bits and $D_{\rm KL,2}$ is KL divergence using base-two logarithms. This decomposes the expected rate estimate into intrinsic uncertainty plus model mismatch; it elaborates source Eqs. 10.6–10.8. It assumes the model gives nonzero probability wherever $q$ does. Training F can reduce mismatch; training A and B also changes $q$. Reducing rate alone may collapse the code to a constant, which is why distortion must remain in the objective.

<details><summary>Predict: if F samples a plausible new index sequence, has it decompressed the original image?</summary>

No. Generation draws a new sequence. Decompression must recover the sequence encoded in the message, using matching probabilities and coding rules.

</details>

## 6 · Count bits carefully

[![Four indices from a four-entry codebook require eight fixed-width bits or 0.5 bits per pixel for sixteen pixels; an illustrative model assigns an ideal five-bit sequence length.](/figures/neural-compression/rate.svg)](/figures/neural-compression/rate.svg)

For $K=2^\kappa$ possible entries, fixed-width coding takes $\kappa=\log_2K$ bits per index. With $M$ indices and $P$ image pixels,

$$
L_{\rm fixed}=M\log_2K,\qquad
\mathrm{bpp}_{\rm fixed}=\frac{M\log_2K}{P},\qquad
\mathrm{bpp}_{\rm ideal}=\frac{-\log_2p_\lambda(i)}{P}.
$$

Here $P$ counts spatial pixels: for RGB it is height × width, not three times that number. All our worked images are grayscale, so input dimension and pixel count coincide. For $M=4$, $K=4$, $P=16$, the fixed payload is 8 bits or 0.5 bpp. Conditional probabilities $(1/2,1/2,1/4,1/2)$ instead give ideal length 5 bits or 0.3125 bpp.

**Correction to the supplied text:** the “Bits per Pixel” box on printed p. 271 (PDF p. 285) correctly starts with $\kappa M$ bits, but then gives a uniform probability $1/(\kappa M)$. That is not the probability of a length-$M$ uniform sequence. The sequence probability is $K^{-M}$, and $-\log_2K^{-M}=M\log_2K$. In our example, the source’s intermediate expression would give 3 bits instead of 8. The correction follows directly from counting $K^M$ possible sequences.

For a deployed codec, measure the **actual full stream length** and include headers, termination, side information and any per-image model updates. Shared pretrained weights and codebooks may be amortized, but state that convention. A favorable likelihood estimate is not a measured file size.

## 7 · The rate–distortion tradeoff is a choice of units and priorities

[![Two constructed reconstruction candidates change order under the objective distortion plus beta times rate: beta 0.005 favors fidelity, beta 0.02 favors fewer bits.](/figures/neural-compression/tradeoff.svg)](/figures/neural-compression/tradeoff.svg)

Compare candidate A with distortion 0.01 and rate 4 bits against B with distortion 0.04 and rate 1 bit. Under $d+\beta r$, $\beta=0.005$ gives scores 0.030 and 0.045, favoring A; $\beta=0.02$ gives 0.090 and 0.060, favoring B. This is a constructed comparison, not evidence of a trained model’s frontier. For this convention, raising $\beta$ makes bits more expensive; alternative papers may put the multiplier on distortion instead.

Distortion direction also matters. MSE is lower-is-better, whereas PSNR and MS-SSIM are quality scores usually reported as higher-is-better. Although §10.2.4 loosely lists them under distortion, do not directly minimize a positive PSNR term. With pixel maximum $a$ and nonzero MSE,

$$
\mathrm{PSNR}=10\log_{10}\frac{a^2}{\mathrm{MSE}}.
$$

For the normalized example, $a=1$ and MSE $=0.04$, so PSNR $\approx13.98$ dB. Using $255$ without rescaling the image/MSE would be inconsistent. Zero MSE gives an infinite ideal PSNR. PSNR summarizes squared error; it does not guarantee perceptual or semantic faithfulness.

## 8 · What the source demonstrates, and what remains untested

[![Original textbook Figure 10.5 on printed page 273 shows distortion and rate versus epochs and four rows of original, reconstruction and sampled digit-like images.](/figures/neural-compression/source-figure-10-5.png)](/figures/neural-compression/source-figure-10-5.png)

This is the **actual supplied PDF page**, not a generated result. Figure 10.5 accompanies the chapter’s $\beta=1$ example. Read A and B as training curves over approximately 200 epochs: distortion and the reported rate decline substantially early, then change more slowly. C contrasts originals, reconstructions and images generated from the autoregressive model’s sampled codes. The generated column illustrates a different path from reconstruction.

The example does **not** supply a named dataset/split, a baseline codec comparison, a complete training protocol or measured file-size accounting in this section. The plotted rate should not be silently relabeled bpp: Listing 10.5 averages a natural-log rate over latent coordinates. We therefore do not invent exact benchmark numbers or a JPEG speed/compression advantage. A one-$\beta$ training trajectory is not a rate–distortion frontier across operating points.

A useful proposed ablation would keep the transform, quantizer and codebook fixed while comparing a uniform, factorized and autoregressive entropy model on the **same held-out indices**. Measure actual bits including common overhead, decoding latency and unchanged reconstruction distortion. That isolates predictability gains. A second experiment could retrain the full system across $\beta$ values and compare matched-bpp distortion; that tests a different question because the representation changes too.

## Reading map and the idea to carry forward

| Question | Supplied source | This guide’s addition |
| --- | --- | --- |
| What must a codec transmit? | §§10.2.1–10.2.4; Fig. 10.1 | Explicit sender/receiver contract |
| Which components are learned? | §10.4.1; Listings 10.1–10.2; Fig. 10.4 | Consistent A–F architecture boxes |
| How does quantization get gradients? | §10.4.2; Eq. 10.3; Listing 10.3 | Soft/hard distinction, ties and formula mismatch |
| Where does the generative model help? | §10.4.3; Listing 10.4 | Sequential decoding trace and entropy/KL identity |
| What does the loss measure? | §10.4.4; Eqs. 10.4–10.9; Listing 10.5 | Explicit bits/nats and reduction conventions |
| How many bits are sent? | p. 271, “Bits per Pixel” | Corrected uniform-sequence calculation |
| What empirical evidence is shown? | §10.4.5; Fig. 10.5, p. 273 | Separate training illustration from codec benchmarking |

**The idea to carry forward:** the generative model supplies probabilities that make a representation cheap to communicate. The encoder decides what to retain; quantization makes symbols; entropy coding makes a message; the decoder turns recovered symbols into an image. Keeping those jobs separate makes both the mathematics and the engineering testable.

Return to the [model-family roadmap](/papers/t2-deep-generative-modeling) for autoregressive likelihoods and VAEs, or the [Topic 2 overview](/papers/t2-generative-overview) to connect compression with the course’s generative-model papers.
