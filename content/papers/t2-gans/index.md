---
title: "Generative Adversarial Nets · learn to generate through a critic"
shortTitle: "GANs"
topic: "topic-2"
order: 4
materialKind: "extra"
description: "Trace the original GAN game through alternating gradients, implicit distributions, a worked update, ideal equilibrium and the limits of its evidence."
status: "ready"
source: "https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html"
pdf: "/papers/generative-adversarial-nets.pdf"
---
## The puzzle: who tells a generator what looks real?

You can write a neural network that turns random numbers into an image. But what loss tells it to produce varied, plausible images when there is no single correct image for each noise draw? Comparing every output with one arbitrary training image would reward the wrong correspondence.

**A GAN learns its training signal: a discriminator distinguishes real examples from generated ones, and a generator learns through that discriminator.** The competition concerns the distribution of outputs, not a prescribed image for each input.

This extra-reading guide follows Goodfellow and colleagues’ **nine-page NIPS 2014 proceedings paper, Generative Adversarial Nets**, supplied as `topic 2/extra reading material/GANs.pdf`. The PDF metadata records modification on December 3, 2014; this is a file timestamp, not an arXiv revision date. [Read the exact course copy](/papers/generative-adversarial-nets.pdf). The [proceedings record](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html) confirms title, authors and venue.

Allow 25 minutes. You need probabilities, natural logarithms and the chain rule. Six mechanism diagrams below are teaching schematics; the seventh plots reported Table 1 numbers. None are generated image samples. Click figures to enlarge them. No trained GAN runs here.

## 1 · Architecture: the discriminator supplies a differentiable signal

[![Noise enters generator B, generated and real vectors enter the same discriminator C, and alternating updates change different parameter sets.](/figures/gans/architecture.svg)](/figures/gans/architecture.svg)

Follow **A → B → C**. Sample noise $z\in\mathbb R^d$ from a prior $p_z$. The generator $G_\theta$ maps it to a data vector $\tilde x=G_\theta(z)\in\mathbb R^D$. The discriminator $D_\phi$ maps either $\tilde x$ or a real vector $x$ to one scalar in $[0,1]$. Here $d$ is noise dimension, $D$ in $\mathbb R^D$ is data dimension, and $D_\phi$ denotes the network, not a dimension. Parameters $\theta$ and $\phi$ belong to different players.

Under the balanced real/fake classification task, the scalar estimates the probability of **real origin**. It is not a semantic class label, a normalized image density, or an objective measure of human-perceived quality. Real and fake samples pass through the same discriminator weights.

The functional architecture follows §3. The experiments (§5) use rectifier and sigmoid generator activations, maxout discriminator activations, and dropout in the discriminator; noise enters the generator only at its bottommost layer. Figure 2 also includes a convolutional discriminator/deconvolutional generator CIFAR-10 variant. The diagrams do not invent layer counts or widths, and do not claim every experimental model has the same backbone.

**Design reasoning, as a teaching interpretation:** if a fixed distance misses what makes data realistic, train a comparison function to detect the generator’s current mistakes. Then use its input gradient to change the generator. This makes the signal adaptive, but also makes the objective move as the opponent learns.

## 2 · The game: two players want opposite outcomes

Before looking at signs, ask what each player should prefer. The discriminator wants real scores near one and fake scores near zero. The generator wants fake scores to increase. Paper Eq. 1 expresses this as:

$$
\min_\theta\max_\phi V(D_\phi,G_\theta)
=\mathbb E_{x\sim p_{\rm data}}[\log D_\phi(x)]
+\mathbb E_{z\sim p_z}[\log(1-D_\phi(G_\theta(z)))].
$$

$p_{\rm data}$ is the target data distribution. Expectations average over real examples and independent noise draws. Logs are natural logs throughout this guide. The two terms meet at the **classification objective box** after C. Maximizing the first rewards accepting real examples; maximizing the second rewards rejecting generated examples. When updating $\theta$, the real term has no dependence on the generator, so only the fake term contributes.

For one real score 0.8 and fake score 0.2, the discriminator value is $\log(0.8)+\log(0.8)\approx-0.4463$. Better classification pushes this value upward toward zero. Conversely, with D fixed, increasing the fake score from 0.2 to 0.4 changes the generator’s minimax term from $\log(0.8)$ to $\log(0.6)$: it decreases, as desired. A smaller generator term and a larger discriminator value are consistent because different players move different parameters.

This is a game, not joint minimization of one loss by both networks. Exact scores zero or one can produce divergent logs; practical implementations must evaluate the equivalent losses stably. The mathematical objective does not by itself specify a numerically stable implementation.

## 3 · Training and sampling follow different paths

[![Three rows separate discriminator updates, generator updates through a frozen discriminator, and generator-only sampling.](/figures/gans/paths.svg)](/figures/gans/paths.svg)

Read the first two rows as a repeated training cycle, and the final row as deployment. Paper Algorithm 1 uses minibatches of size $m$, $k$ discriminator steps per generator step, and fresh noise for the generator step. The authors report $k=1$ and momentum in their experiments.

1. Draw real examples and noise; compute generated examples.
2. Ascend the minibatch estimate of $V$ with respect to $\phi$, holding $\theta$ fixed.
3. Draw fresh noise; descend the generator objective with respect to $\theta$, holding $\phi$ fixed.
4. Repeat, carrying both networks’ updated weights and optimizer state forward. Noise draws and minibatches are replaced.

**Fixed discriminator weights do not mean a blocked discriminator gradient.** During step 3 the gradient must pass from its output back to its input $G_\theta(z)$ and then to $\theta$. Detaching the generated sample in this step would remove that learning path. During the discriminator step, the generator is not updated.

Algorithm 1 writes the original minimax generator objective; §3 separately introduces the practical non-saturating alternative explained below. Its noise-prior lines print $p_g(z)$, whereas §3 calls the noise prior $p_z(z)$ and reserves $p_g$ for the output distribution. We use $p_z$ consistently to avoid confusing these objects.

After training, **sample $z$, run $G_\theta(z)$, return the output**. No discriminator, encoder, real reference image or Markov chain is needed for this unconditional sampling path. Independent noise draws give independent outputs for a fixed deterministic generator, but independence does not guarantee diverse outputs.

<details><summary>Predict: if D’s weights are frozen during the generator step, can D still teach G?</summary>

Yes. Its input derivative remains available. Freezing which parameters the optimizer changes is different from stopping differentiation through the function.

</details>

## 4 · Representation: a map induces a distribution

[![A fixed generator moves noise probability mass to output space; a square mapping concentrates mass and a constant mapping collapses all noise to one output.](/figures/gans/representation.svg)](/figures/gans/representation.svg)

Read the rows as a general construction, a scalar example, and a failure case. The output distribution $p_g$ is **implicit**: it is defined by drawing $z\sim p_z$ and applying G. You can sample without evaluating $p_g(x)$.

For a constructed example, let $z$ be uniform on $[0,1]$ and $G(z)=z^2$. Then

$$
\Pr(G(z)\leq 1/4)=\Pr(z\leq 1/2)=1/2.
$$

Half the input mass lands in the first quarter of output space. Uniformly spaced noise coordinates need not become uniformly spaced outputs. For $0<x<1$, differentiating the cumulative probability $\sqrt{x}$ gives output density $p_g(x)=1/(2\sqrt{x})$. This tractable one-dimensional derivation illustrates Figure 1’s mass-transport intuition; a general neural generator need not provide this density formula or be invertible.

If instead $G(z)=c$ for every z, all noise draws yield the same output c. This is an extreme diversity failure. Section 6 calls a related collapse of many noise values to the same output the “Helvetica scenario.” A realistic-looking sample does not establish coverage of the target distribution. Likewise, interpolations in latent space can reveal continuity without proving that all data modes are represented.

## 5 · Why change the generator loss?

When the discriminator confidently rejects generated samples, the original minimax objective can give a weak generator signal. Section 3 therefore proposes maximizing $\log D(G(z))$, equivalently minimizing the **non-saturating** loss:

$$
L_G^{\rm NS}=-\mathbb E_{z\sim p_z}\log D_\phi(G_\theta(z)).
$$

This loss occupies the same box after C and follows the same backward path to B. It changes the learning signal, not the sampling architecture.

[![At fake scores 0.01, 0.5 and 0.99, minimax logit derivatives are minus the score while non-saturating derivatives are the score minus one.](/figures/gans/gradients.svg)](/figures/gans/gradients.svg)

The diagram is a teaching calculation, not measured training performance. Let $a$ be the fake-sample discriminator logit and $s=\sigma(a)=1/(1+e^{-a})$ its score. Since $ds/da=s(1-s)$, the chain rule gives:

$$
\frac{\partial\log(1-s)}{\partial a}=-s,
\qquad
\frac{\partial[-\log s]}{\partial a}=s-1.
$$

At $s=0.01$, the magnitudes are 0.01 and 0.99. Both signs encourage increasing the logit under gradient descent, but the second has a much stronger signal when D rejects the sample. These are derivatives **with respect to the logit**, not directly with respect to the score or the generator’s weights. The full generator gradient also includes the discriminator’s input Jacobian and the generator’s parameter Jacobian.

The paper says the alternative has the same fixed point and stronger early gradients. This does not make it the same objective everywhere, nor does it remove all causes of unstable training. In particular, the minimax Jensen–Shannon identity in §7 below is not an identity for this replacement loss.

## 6 · Worked update: follow a number through the architecture

[![A toy scalar generator produces zero, a fixed sigmoid discriminator scores it 0.1192, and a chain-rule update increases its parameter to 0.1762.](/figures/gans/worked.svg)](/figures/gans/worked.svg)

This is one constructed generator step, not an experimental result or a complete GAN training run. Choose $G_\theta(z)=\theta z$, $D(x)=\sigma(2x-2)$, $z=1$, and initial $\theta=0$. The discriminator parameters 2 and −2 are fixed for this step.

**A → B:** noise 1 produces $x=0$. **B → C:** the logit is $a=-2$, so $D(x)\approx0.1192$. **Loss box:** $L=-\log(0.1192)\approx2.1269$.

**C → B, backward:** combine the three local derivatives:

$$
\frac{\partial L}{\partial\theta}
=\underbrace{(D(x)-1)}_{\text{loss to logit}}\quad
\underbrace{2}_{\text{logit to image}}\quad
\underbrace{z}_{\text{image to parameter}}
\approx-1.7616.
$$

For learning rate $\eta=0.1$, gradient descent gives

$$
\theta_{\rm new}=\theta-\eta\frac{\partial L}{\partial\theta}
=0-0.1(-1.7616)\approx0.1762.
$$

Reusing the same z only to check the arithmetic gives $x_{\rm new}\approx0.1762$, logit −1.6477, score approximately 0.1614 and loss approximately 1.8237. The generator improved its score under this fixed discriminator. This alone says nothing about image quality or distribution coverage. A real training iteration would also update D and draw new minibatches.

<details><summary>Predict: what happens to this generator update if you detach x before passing it into D?</summary>

The dependency from the loss back to θ is broken. The useful −1.7616 gradient cannot reach the generator, even though D still computes a score.

</details>

## 7 · Ideal equilibrium: why one half can mean success

A strong discriminator compares the local prevalence of real and generated samples. With G fixed, paper Eq. 2 gives

$$
D_G^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_g(x)}.
$$

The star denotes an optimal discriminator for that particular generator. At points with positive total density, this ratio follows by maximizing $a\log y+b\log(1-y)$ with nonnegative $a=p_{\rm data}(x)$ and $b=p_g(x)$. Its derivative $a/y-b/(1-y)$ vanishes at $y=a/(a+b)$ when both densities are positive; one-sided cases give boundary optima. Where both densities vanish, the objective does not constrain D. Balanced origin sampling matters: different class priors change the ratio.

[![Two-bin distributions give optimal scores 0.75 and 0.25 when mismatched, and 0.5 in both bins when matched; the theorem requires ideal conditions.](/figures/gans/equilibrium.svg)](/figures/gans/equilibrium.svg)

Read the first row as a distribution-level toy example, not a differentiable discrete-output training recipe. With data masses $(0.75,0.25)$ and generator masses $(0.25,0.75)$, the ideal discriminator is $(0.75,0.25)$. The first bin is underproduced; the second is overproduced. At matching masses the scores become $(0.5,0.5)$.

Substituting the ideal discriminator into the minimax value gives paper Eqs. 5–6:

$$
C(G)=\max_D V(D,G)=-\log4+2\operatorname{JSD}(p_{\rm data}\Vert p_g),
$$

$$
\operatorname{JSD}(p\Vert q)=\tfrac12\operatorname{KL}(p\Vert M)+\tfrac12\operatorname{KL}(q\Vert M),
\qquad M=\tfrac12(p+q).
$$

Here KL is the Kullback–Leibler divergence, $\operatorname{KL}(p\Vert q)=\int p(x)\log[p(x)/q(x)]\,dx$ for densities, with a sum for discrete masses. $M$ is the mixture distribution. JSD is nonnegative and zero exactly when the distributions agree, so the ideal minimum is $-\log4\approx-1.3863$. Our mismatched two-bin example gives $C(G)\approx-1.1247$ and JSD approximately 0.1308; matching gives JSD zero.

This mathematics describes the **ideal discriminator box and the induced distribution**, not a computable GAN likelihood. Proposition 2 assumes enough capacity, an optimal discriminator at each step, and sufficiently small improvements in distribution space. The paper explicitly says its proof does not apply to finite neural networks optimized through their parameters. A flat score of one half from an untrained or weak D is therefore not evidence that G has learned the data.

<details><summary>Predict: does D(x)=0.5 prove a successful generator?</summary>

Only with the crucial optimal-discriminator qualification on the relevant support. A discriminator that ignores its input also outputs one half while telling you nothing about distribution matching.

</details>

## 8 · What the experiments establish

[![Two panels reproduce Table 1 mean Parzen log-likelihood estimates and standard errors for DBN, stacked CAE, deep GSN and adversarial nets on MNIST and Toronto Face Database.](/figures/gans/results.svg)](/figures/gans/results.svg)

This figure redraws **measured Table 1 values from the supplied proceedings PDF**. Each panel starts at zero but uses a different scale; compare models within a dataset. Thin white marks show the reported standard errors. Higher estimates are better under this evaluation procedure.

| Model | Real-valued MNIST | Toronto Face Database |
| --- | ---: | ---: |
| DBN | 138 ± 2 | 1909 ± 66 |
| Stacked CAE | 121 ± 1.6 | 2110 ± 50 |
| Deep GSN | 214 ± 1.1 | 1890 ± 29 |
| Adversarial nets | **225 ± 2** | 2057 ± 26 |

Section 5 fits a **Gaussian Parzen density estimator to generated samples**, selects its kernel width using validation data, then evaluates test examples under that fitted density. These are not exact log-likelihoods under $p_g$. For MNIST the standard error is computed across test examples; for TFD it is computed across folds, with width selected for each fold. Neither is a measure of variability across independent GAN training seeds. These MNIST comparisons use real-valued data, not the binary version.

GANs exceed the listed MNIST baselines on this estimate, but stacked CAE has the higher TFD estimate. The authors themselves note high variance and poor behavior of Parzen evaluation in high dimensions. A favorable fitted-density score does not establish universally superior generation or coverage.

The experiments include CIFAR-10, but Table 1 gives no CIFAR-10 likelihood row. Source Figure 2 (PDF p. 7) shows random generated samples and neighboring training examples; Figure 3 shows latent interpolations. The authors state that the displayed samples are not cherry-picked. The nearest-neighbor comparison is evidence relevant to memorization, not a comprehensive proof that no training information is memorized. They explicitly avoid claiming their samples are better than existing methods’ samples.

### What would a useful follow-up test isolate?

These are proposed experiments, not findings from the source. Hold model capacity, data split and computation fixed when comparing minimax and non-saturating losses; measure both learning signal and sample coverage. Repeat training with multiple seeds to assess synchronization failures. Test whether increasing discriminator capacity helps or merely overfits finite training examples. Pair image inspection with a coverage assessment rather than treating a few sharp outputs as a distributional guarantee.

Section 6 identifies the missing explicit density and the need to synchronize G and D as disadvantages. The basic gradient path also assumes differentiability through generated outputs; sampling hard discrete symbols breaks that straightforward path. These statements describe the original setup and do not claim later GAN variants cannot address such limitations.

## 9 · The idea to carry forward

**A learned comparison task can train a sampler without evaluating its output density.** Its power comes from adapting the signal to current mistakes; its difficulty comes from coordinating two changing networks.

Compare this with the [Deep Generative Modeling roadmap](/papers/t2-deep-generative-modeling): a VAE’s second network approximates latent inference, whereas a GAN’s second network distinguishes origins. Compare it with [Flow Matching](/papers/t2-flow-matching): flow matching uses constructed regression targets along a chosen path; the original GAN learns a competing discriminator and samples with a single generator pass. Single-pass sampling does not imply easy training.

Redraw A → B → C, mark which weights move in each training step, and then cross out C for sampling. Explain why an ideal D returning one half is good news, while an arbitrary D doing so is inconclusive. Those distinctions connect architecture, gradients and the theorem.

## Sources and reading map

- [Exact local course PDF](/papers/generative-adversarial-nets.pdf), Goodfellow et al., *Generative Adversarial Nets*, NIPS 2014, nine pages. All experimental values and mechanism claims above follow this copy.
- [Official proceedings identity](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html). The course filename “GANs” refers to this original paper, not a survey of subsequent variants.
- **§3, Eq. 1, p. 3:** game, functional architecture and non-saturating alternative. **Figure 1 and Algorithm 1, p. 4:** mass transport and alternating updates.
- **§4, Eqs. 2–6, pp. 4–5:** optimal discriminator, JSD and theorem assumptions. **§5/Table 1, p. 6; Figures 2–3, p. 7:** experiments and evidence.
- **§6, p. 6; §7, p. 7; Table 2, p. 8:** limitations and proposed extensions. Conditional generation and auxiliary inference are future directions here, not prerequisites for the unconditional architecture.

All scalar networks, derivative tables and two-bin calculations are original teaching examples. They are separated from the paper’s reported experiments.
