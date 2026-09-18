---
title: "Generalization and learning dynamics"
shortTitle: "Generalization and learning dynamics"
description: "Six papers on why neural networks learn and generalize. Start with the information bottleneck and its measurement caveats."
order: 1
status: "exploring"
tags: []
locale: "en"
---
## Learning path

For shared foundations, read the [Deep representation learning roadmap](/papers/representation-overview): the general March 2026 textbook connects PCA, denoising, coding-rate objectives, unrolled networks and consistency. Its eight recall cards are grouped here for study and printing; the book remains general course material in the library.


Start with [Information bottleneck](/papers/t1-paper-1): follow classifier training, whole-layer information measurements, and the compression–prediction tradeoff through seven diagrams and eight recall cards. The guide follows the supplied April 2017 v3 paper and separates reported experiments from toy examples and theoretical interpretation.

Continue with [Nonvacuous generalization bounds](/papers/t1-paper-2): turn a trained network into a distribution of networks, then trace the PAC-Bayes objective and final certificate through eight diagrams and eight recall cards. The guide follows the supplied October 2017 v2 PDF and distinguishes randomized-network guarantees from deterministic test accuracy.

Then explore [A Bayesian perspective on SGD](/papers/t1-paper-3): connect prior-relative width and evidence to mini-batch gradient noise, finite-batch limits and momentum through seven diagrams and eight cards. The guide uses the supplied ICLR 2018 conference PDF and distinguishes the Bayesian sampling analogy from an exact sampling claim.

Continue with [Deep double descent](/papers/t1-paper-4): follow the fitting threshold through model width, training time and dataset size, with eight teaching diagrams, a source result figure and eight cards. The guide uses the updated December 2021 journal paper and records its metric and training-budget ambiguities.

Then study [Understanding grokking](/papers/t1-paper-5): trace delayed generalization through equal embedding sums, constraint rank, spectral timescales and decoder competition. Seven teaching diagrams, the supplied Figure 4 and eight cards distinguish the toy theory from transformer evidence and explain the missing appendices and normalization caveats.

Finish with [Binarized networks and compression](/papers/t1-paper-6): separate cross-entropy training from weight diagnostics, compute block entropy and BDM, and examine the limits of loss correlations. Seven reproducible diagrams, including a measured interval plot, and eight cards follow the supplied May 2026 paper. All six supplied Topic 1 papers now have guides.

**Working topic label:** inferred from the available papers, not verified against a course syllabus.

## Textbook foundations

The [Understanding Deep Learning roadmap](/papers/understanding-deep-learning) follows Prince’s supplied November 21, 2024 copy from predictions and likelihoods through gradients, evaluation, attention and diffusion. It remains general course material; eight foundation cards are grouped with Topic 1 for practice and printing.

## A unifying position

[Deep Learning is Not So Mysterious or Different](/papers/not-so-mysterious) connects soft inductive biases, countable-hypothesis bounds and double descent. The July 10, 2025 overview remains general course material; its eight cards are grouped here for practice and printing. Worked examples and source Figure 5 separate mathematical guarantees from synthetic evidence and open questions.

## Selection and uncertainty foundations

[Selection under uncertainty](/papers/selection-under-uncertainty) follows Seldin’s January 2026 textbook from K-NN validation through finite-class and PAC-Bayes guarantees to online learning. Seven reproducible diagrams and eight foundation cards connect source assumptions to worked examples. The selective roadmap remains general course material in the library.
