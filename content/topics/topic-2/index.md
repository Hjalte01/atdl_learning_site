---
title: "Deep generative modeling"
shortTitle: "Deep generative modeling"
description: "Six papers on generative models. Explore illustrated guides for all six papers, including Qwen-Image."
order: 2
status: "active"
tags: ["generative modeling"]
locale: "en"
---
## Learning path

Start with [FLUX.1 Kontext](/papers/flux-kontext). Learn why the reference image stays fixed while the output is generated from noise, and how both streams interact. Then use the study cards to check the distinctions.

Continue with [DiffAtlas · paper 4](/papers/t2-paper-4): follow the architecture, trace the equations through image–mask diffusion, and explore why inference replaces the image with a noisy patient scan.

Continue with [REPA · paper 1](/papers/t2-paper-1): see how a clean-image teacher guides early generator features during training, then disappears from the sampling path. Trace the two losses and examine what the training speedup actually measures.

Continue with [Mean Flows · paper 2](/papers/t2-paper-2): learn why an interval-average velocity supports one-call generation, derive the JVP target, and distinguish sampling steps from network evaluations.

Continue with [DiME · paper 3](/papers/t2-paper-3): trace clean-image classifier gradients back into a noisy diffusion state, examine the gradient approximation, and question what counterfactual metrics actually establish.

Continue with [Qwen-Image · paper 6](/papers/t2-paper-6): follow text rendering through the data curriculum, VAE, and flow model; trace dual-reference conditioning for editing and inspect where benchmark gains still leave failures.
