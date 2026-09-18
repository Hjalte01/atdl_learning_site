---
title: "Understanding Deep Learning: a reading roadmap"
shortTitle: "Understanding Deep Learning"
topic: "general"
order: 2
materialKind: "overview"
description: "Connect predictions, losses, gradients and evaluation, then follow attention and diffusion into the course readings."
status: "ready"
pdf: "/papers/understanding-deep-learning.pdf"
---
## The puzzle: a good fit to what?

A network predicts every training label correctly. Has it learned the task, memorized the examples, or exploited a shortcut? Training accuracy cannot decide. To investigate, we need to distinguish the function a network computes, the loss used to fit it, the algorithm that changes its parameters, and the evidence used to judge it.

This roadmap follows **Simon J. D. Prince’s Understanding Deep Learning, supplied copy dated November 21, 2024**. The cover states that MIT Press released the final version in December 2023; the date here identifies the supplied document, not a new edition claim. It is general course material, not a numbered topic paper. This is a selective reading route through a textbook, not an exhaustive chapter summary. Allow about 20 minutes, then use the source map to continue reading.

All diagrams and numerical examples below are original teaching schematics and calculations. No trained model runs on this page, and no toy value is a benchmark result. A useful interpretation of the book’s progression is: first specify a prediction, then justify its objective, learn to optimize it, and only then ask whether it transfers. That is a teaching interpretation, not a claim about the author’s private reasoning.

## 1 · Choose a route through the book

[![Three reading routes progress from foundations through architectures to generative models and broader questions.](/figures/understanding-deep-learning/map.svg)](/figures/understanding-deep-learning/map.svg)

Read each row as a suggested route, not as a claim that every chapter depends on every earlier one. Chapters 2–9 establish the common training vocabulary. Chapters 10–13 adapt computation to images, sequences and graphs. Chapters 14–18 examine generation; Chapters 19–21 cover reinforcement learning, explanations of deep learning’s success and ethics.

| Prerequisite | Where it becomes useful |
| --- | --- |
| Vectors, matrices and dimensions; Appendix B | Layers in Chapter 4, attention in Chapter 12. |
| Derivatives and the chain rule; Appendix B.5 | Gradient descent and backpropagation in Chapters 6–7. |
| Probability, likelihood and expectations; Appendix C | Losses in Chapter 5 and generative models in Chapters 14–18. |
| Separate training, validation and test data | Assessing the research claims in Chapters 8–9 and the course papers. |

Start with Chapters 2–7 if the distinction between a network and its optimizer is unfamiliar. Read Chapters 8–9 before interpreting training curves. For Topic 2, continue through Chapter 14’s evaluation questions before selecting a generative family.

## 2 · Follow one prediction through a network

A linear model can move and tilt a line, but cannot represent every nonlinear relation. Chapters 3–4 introduce nonlinear hidden units and their composition. The hidden representation is an intermediate computation learned for the objective; it is not automatically a human-interpretable concept.

[![A two-dimensional input becomes two ReLU activations, then one logit and a Bernoulli probability; the label enters only at the training loss.](/figures/understanding-deep-learning/network.svg)](/figures/understanding-deep-learning/network.svg)

Follow the upper row through the model; the lower row adds the probability interpretation and training target. This is a complete **toy binary classifier**, not a prescribed architecture from the book. Using the matrix-layer convention of §4.4, write

$$
a=Wx+b,\qquad h=\operatorname{ReLU}(a),\qquad z=v^\top h+c,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Here $x\in\mathbb R^2$ is the input, $W\in\mathbb R^{2\times2}$ and $b\in\mathbb R^2$ define the hidden affine map, and $h\in\mathbb R^2$ is the representation. ReLU replaces negative coordinates with zero. The output weights $v\in\mathbb R^2$ and scalar bias $c$ produce the scalar **logit** $z$. The sigmoid converts it to a Bernoulli parameter $p$, interpreted as the model’s probability of label 1. Collect the learned weights and biases as $\theta=(W,b,v,c)$; the book generally uses $\phi$ for parameters.

**Work the boxes.** Take $x=(1,2)^\top$, $W=\left[\begin{smallmatrix}1&-1\\0&1\end{smallmatrix}\right]$, $b=(0,0)^\top$, $v=(1,1)^\top$, and $c=-1$. Then $a=(-1,2)^\top$, $h=(0,2)^\top$, $z=1$, and $p\approx0.7311$. A threshold of 0.5 predicts class 1. This does not establish that 73.11% of similar real cases have that label; calibration requires evaluation.

ReLU makes the hidden map nonlinear; stacking affine maps without nonlinearities would still produce an affine map. The zero first coordinate is a consequence of this input and these parameters, not proof that the first feature is permanently useless.

## 3 · Choose a loss by specifying an output distribution

A prediction needs a criterion for being good. Chapter 5’s recipe chooses a distribution over possible outputs, makes its parameters depend on the network, and minimizes negative log-likelihood. This connects the objective to an explicit modeling assumption.

For binary labels $y\in\{0,1\}$, the Bernoulli likelihood is $p^y(1-p)^{1-y}$. The per-example binary cross-entropy, corresponding to one term of book Eq. (5.20), is

$$
\ell(\theta;x,y)=-y\log p-(1-y)\log(1-p).
$$

The logarithms are natural. $p=\sigma(f_\theta(x))$ comes from the **probability box**; the observed $y$ enters the **loss box**. Over independent training examples, log-likelihoods add; a mean instead of a sum changes the gradient scale. This loss rewards probability assigned to the observed outcome, rather than just whether the thresholded label is correct.

For the same $z=1$ above, label $y=1$ gives $\ell=-\log(0.7311)\approx0.3133$. If the label is instead 0, the loss is $-\log(0.2689)\approx1.3133$. Thresholding before computing the loss would discard this probability information and obstruct useful gradients.

[![Identical logits feed probabilities, label-specific losses and logit derivatives; a separate regression row links a Gaussian assumption to squared error.](/figures/understanding-deep-learning/loss.svg)](/figures/understanding-deep-learning/loss.svg)

Read the first two rows as alternative observed labels for the **same** prediction. The third row is a different task: §5.3 derives squared-error regression from a Gaussian output with fixed variance. It is not another stage of the classifier. All displayed values are calculated teaching examples.

For fixed Gaussian variance $s^2>0$ and predicted mean $\mu=f_\theta(x)$, negative log-likelihood is $(y-\mu)^2/(2s^2)$ plus a parameter-independent constant. Thus least squares has a probabilistic motivation, but it does not make every dataset Gaussian or correctly account for input-dependent noise. The book contrasts fixed and input-dependent variance in Figure 5.5.

<details><summary>Predict: two classifiers both predict label 1, with probabilities 0.51 and 0.99. Must they have the same loss?</summary><p>No. If the label is 1, the second receives a smaller cross-entropy loss. If the label is 0, its confident mistake receives a much larger loss. Accuracy hides that distinction.</p></details>

## 4 · Separate backpropagation from the parameter update

Backpropagation computes how the loss changes with each parameter by reusing chain-rule calculations. An optimizer then uses those derivatives to change the parameters. These are different operations (Chapters 6–7).

[![Training stores a forward computation, propagates loss derivatives backward and updates parameters; inference holds learned parameters fixed and needs no target.](/figures/understanding-deep-learning/paths.svg)](/figures/understanding-deep-learning/paths.svg)

Read the two rows independently. Training repeats over data or minibatches; inference changes the input while holding learned parameters fixed. This schematic omits architecture-specific behavior such as dropout and batch-normalization training modes, discussed in Chapters 9 and 11.

For our sigmoid classifier, the chain rule simplifies the derivative at the **logit arrow** to

$$
\frac{\partial\ell}{\partial z}=p-y,\qquad
\frac{\partial\ell}{\partial v}=(p-y)h,\qquad
\frac{\partial\ell}{\partial c}=p-y.
$$

These are teaching derivations for the displayed model, not numbered equations quoted from the book. The hidden derivative is $\partial\ell/\partial a=(p-y)v\odot\mathbf1_{a>0}$, where $\odot$ means coordinatewise multiplication and the indicator is 1 at positive pre-activations. ReLU’s derivative at exactly zero needs a convention; our example avoids that point.

Gradient descent changes parameters with

$$
\theta_{k+1}=\theta_k-\eta\nabla_\theta L(\theta_k).
$$

This is book Eq. (6.3) with renamed symbols: $k$ counts updates, $L$ is the selected training loss, and $\eta>0$ is the learning rate. The gradient is evaluated at the current parameters. A step that is too large can raise the loss; a zero gradient need not identify a global minimum.

**Trace one partial update.** Freeze $W,b$ solely to keep the arithmetic short, use the single example above with $y=1$, and choose $\eta=0.1$. Then $p-y=-0.26894$, $\partial\ell/\partial v=(0,-0.53788)^\top$, and $\partial\ell/\partial c=-0.26894$. Updating $v,c$ together gives $v'=(1,1.05379)^\top$, $c'=-0.97311$, $z'=1.13447$, and loss approximately **0.2788**. Ordinary full-network training can also update $W,b$; freezing them is a teaching assumption, not the book’s general algorithm.

[![Three repeated output-head updates increase the correct-label probability and decrease the one-example loss while the hidden representation stays frozen.](/figures/understanding-deep-learning/updates.svg)](/figures/understanding-deep-learning/updates.svg)

Read left to right: each state is recomputed from the preceding update, using the same fixed example and learning rate. The declining loss is a toy optimization trace, not a test-performance curve. In stochastic gradient descent, the next minibatch usually changes too, so observed batch losses need not decrease monotonically.

<details><summary>Predict: does a backward pass alone train the model?</summary><p>No. It computes derivatives. The optimizer must apply an update. During ordinary inference, the trained parameters stay fixed and a target label is unnecessary.</p></details>

## 5 · Measure the claim you actually want to make

Chapters 8–9 distinguish fitting from generalization and discuss capacity, regularization and double descent. A model-selection decision is itself informed by data, even when it does not use a gradient.

[![Training data fit parameters, validation data choose a candidate, and a reserved test set evaluates the selected procedure.](/figures/understanding-deep-learning/evaluation.svg)](/figures/understanding-deep-learning/evaluation.svg)

Read left to right as a workflow from §8.5. The arrows describe decisions; they do not move validation or test labels into training. The lower row lists three different questions. This schematic contains no measured results.

The book’s Figure 2.2 offers a deliberately simpler checkpoint: three lines fitted to the same **12 training pairs** have summed squared residuals **7.07, 10.28 and 0.20**. The last line fits that training set best; this is an illustrative regression comparison, not a held-out benchmark or proof of superior generalization. Before trusting a new model, specify the dataset, held-out split, metric and model-selection procedure. A claim about uncertainty or significance also needs repeated trials or an appropriate uncertainty analysis.

For the course, continue from §8.4 to [Deep double descent](/papers/t1-paper-4), and from Chapters 6–9 to [Bayesian SGD](/papers/t1-paper-3) and [generalization bounds](/papers/t1-paper-2). These papers ask different questions about optimization and generalization; a lower training objective does not settle them all.

A proposed ablation might replace the nonlinear hidden layer with an affine one while keeping data splits, search budget and evaluation fixed. It would test the contribution of that design under the chosen protocol. Changing both architecture and training budget would confound the interpretation. These are suggested experiments, not experiments performed by this page.

## 6 · A representation can route information

Chapter 12 extends the layer vocabulary with self-attention. Each token is transformed into a query, key and value. Query–key comparisons decide how much of each value to combine; the routing weights depend on the input.

[![One query compares with two keys, softmax normalizes the scores, and weighted values produce a contextual token representation.](/figures/understanding-deep-learning/attention.svg)](/figures/understanding-deep-learning/attention.svg)

Read the top row as a representation close-up for **one output token**. The lower row supplies a scalar numerical example. This is basic single-head attention, not a full transformer: positional information, multiple heads, residual paths, normalization and feedforward layers require the subsequent sections.

Following Eqs. (12.2)–(12.5), with simplified notation,

$$
q_n=W_qx_n+b_q,\quad k_m=W_kx_m+b_k,\quad v_m=W_vx_m+b_v,
\qquad a_{mn}=\frac{\exp(k_m^\top q_n)}{\sum_{j=1}^{N}\exp(k_j^\top q_n)},
\qquad o_n=\sum_{m=1}^{N}a_{mn}v_m.
$$

There are $N$ input tokens $x_m\in\mathbb R^D$. Queries and keys have a common dimension $d_k$, values and outputs have dimension $d_v$, and the learned matrices and biases have the matching shapes. For fixed output index $n$, normalization runs over source tokens $m$. The book introduces scaling by $\sqrt{d_k}$ subsequently in §12.3; the displayed expression follows its initial unscaled mechanism.

Let $q=1$, keys $(0,\log3)$ and values $(2,6)$, all scalar. Scores are $(0,\log3)$; softmax weights are $(1/4,3/4)$; the output is $0.25(2)+0.75(6)=5$. These are assumed projected quantities, not learned words. Values carry the mixed content; keys are not substituted for values. The weighted sum is linear in values for fixed weights, but the overall mechanism is nonlinear because the weights depend on inputs. An attention weight alone is not a causal explanation of a model decision.

## 7 · Generation changes the object being learned

The classifier modeled a label conditional on an input. Generative models learn a distribution of data, possibly conditional on a prompt or observation. Chapter 14 compares goals and evaluation; Chapters 15–18 cover GANs, normalizing flows, VAEs and diffusion. Their objectives and inference procedures differ.

In Chapter 18, a prespecified forward process corrupts data; a learned reverse process supports generation. Book Eq. (18.7) lets training sample a noisy state directly:

$$
z_t=\sqrt{\alpha_t}\,x+\sqrt{1-\alpha_t}\,\epsilon,
\qquad \alpha_t=\prod_{s=1}^{t}(1-\beta_s),\qquad \epsilon\sim\mathcal N(0,I).
$$

$x$ is clean data, $z_t$ a same-shaped noisy state, $\beta_s\in[0,1]$ the noise schedule, and $I$ identity covariance. **The book’s $\alpha_t$ is cumulative**; many paper guides write this quantity as $\bar\alpha_t$. The square-root factors weight variances under independent noise. With scalar $x=2$, $\alpha_t=0.25$ and chosen noise realization $\epsilon=0$, $z_t=1$; the conditional variance is still $0.75$. That selected realization does not remove uncertainty from the distribution.

The training path is **data + noise + time → network prediction → loss → parameter update**. The generation path is **initial noise → repeated reverse updates with fixed parameters → sample**. Unlike classifier inference, the generated state changes repeatedly. The forward equation alone is not a reverse sampler: §§18.3–18.6 supply the missing reverse-model, loss and implementation choices. Finite corruption schedules generally approach rather than exactly reach a standard normal unless the retained signal vanishes.

Continue to the [generative reading overview](/papers/t2-generative-overview), then [DiffAtlas](/papers/t2-paper-4) to see how observed information constrains generation. Read [Mean Flows](/papers/t2-paper-2) for a different time-dependent construction; do not transfer time conventions or sampling equations just because both start from noise.

## 8 · The idea to carry forward

For each new paper, fill in five blanks: **input, representation, output distribution, training update, evaluation**. Draw training and inference separately. Ask which quantity changes along each arrow and what the reported evidence can establish.

Chapter 20 returns to open explanatory questions; Chapter 21 considers consequences and responsibilities beyond prediction metrics. The roadmap’s worked calculations demonstrate mechanisms, not a universal account of why deep learning works or why a deployed system is suitable for a task.

The eight recall cards are grouped with Topic 1’s foundations for study and printing. The catalog continues to classify this book as general material. No custom simulator is necessary: each numerical example can be checked directly.

## Sources and reading map

[Open the supplied November 21, 2024 textbook](/papers/understanding-deep-learning.pdf). References below use printed pages, not PDF viewer indices. The source’s introductory figures are educational examples; this roadmap does not reproduce or audit the research benchmarks cited throughout the book.

| Source location | What to read for |
| --- | --- |
| Cover, preface, contents and §1.7 | Version identity, scope, prerequisites and reading strategy. |
| §2.2, pp. 18–21, Eqs. 2.4–2.5 and Fig. 2.2 | Model versus loss; the three illustrative training-fit values. |
| Chapters 3–4, especially §4.4, pp. 48–49 | Nonlinear hidden representations and matrix shapes. |
| §§5.2–5.5, pp. 60–69, Eqs. 5.16–5.20 | Distribution-based losses, Gaussian regression and Bernoulli classification. |
| §§6.1–6.2, pp. 77–85; §7.4, pp. 103–107 | Parameter updates, minibatches and reusable backward derivatives. |
| §§8.4–8.5, pp. 127–133; Chapter 9 | Generalization, double descent, validation and regularization. |
| §§12.2–12.4, pp. 208–216, Eqs. 12.2–12.5 | Attention routing, scaling and full-layer extensions. |
| Chapters 14–18; §18.2.1, p. 352, Eq. 18.7 | Generative families and direct forward-noise sampling. |
| Chapters 19–21 | Further routes: reinforcement learning, open theory and ethics. |

The classifier parameters, frozen-head updates, attention numbers and scalar diffusion realization are original teaching choices. They are not the book’s experimental architectures or reported outcomes.
