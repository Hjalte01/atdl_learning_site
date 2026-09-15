# Create an engaging paper learning site

## Interpret the request

“Create sites for topic 2-4” means **Topic 2, paper 4**, not Topics 2 through 4. Resolve `topic` and `order` in `content/papers/*/index.md`; confirm the title against the local course PDF. If the user explicitly asks for a range or all six papers, follow that scope. Build each requested guide to completion; do not fill other papers with invented explanations.

Use the DiffAtlas guide (`content/papers/t2-paper-4/index.md`) as the architecture-to-math teaching reference. Preserve existing guides. This Astro site uses Markdown, KaTeX, custom Astro components, and local SVG/canvas visuals.

## Source before storytelling

1. Read applicable AGENTS.md instructions and the existing site structure.
2. Find the course PDF in the parent topic directory. Record its exact version/date; read the method, equations, figures, experiments, and limitations.
3. Verify primary-source identity online when needed. Prefer the supplied course version. Never mix result numbers from different versions without explanation.
4. Keep a reading map to sections, equation numbers, figures, and tables. Separate reported claims, mathematical derivations, pedagogical intuition, and proposed experiments.
5. Do not invent backbone layers, dimensions, training schedules, or decoding rules that the source does not document. A functional architecture is better than a fabricated detailed architecture.

## Teach visually, with math inside the mechanism

Start with a concrete puzzle and why the obvious approach has a limitation. Explain how the design responds to that limitation. Reconstruct a plausible reasoning path, labeled as interpretation rather than the authors’ private thoughts.

Create multiple useful visuals throughout the explanation—normally at least six for a full paper. Include:

- The problem or contrasting paradigms.
- The full architecture, with labeled tensors and arrows.
- Separate training and inference paths: what is learned, fixed, replaced, sampled, and carried forward?
- A representation close-up (e.g. latent, token, mask, distance field).
- A worked numerical example traced through the same architecture boxes.
- The key iterative process or algorithm over time.
- A result figure where the paper supplies suitable quantitative evidence.

Use consistent colors and symbols across diagrams and equations. Each figure needs a descriptive alt text, a reading explanation, and a statement of what is schematic versus measured. Native SVG is preferred for precise architecture and math; canvas is suitable for interactive numeric illustrations. Use the imagegen skill for tasks that actually need generated raster illustration. Never present a generated image as a paper result. Store local assets under `public/figures/<paper>/` and keep any generation script reproducible.

For each major equation: state the intuition first, define every symbol, identify the exact architecture box or arrow where it operates, work a small example when useful, explain why the operation is chosen, and state its limitations. Distinguish a full algorithm from an illustrative identity. Keep paper equation numbering separate from teaching derivations.

Add an interactive lab only when changing a control reveals something useful. Compute the displayed numbers and visuals from the same state. Label toy assumptions, clipping, fixed random seeds, endpoint limits, and whether an actual trained model is running. Make controls keyboard accessible and useful without animation; provide a static fallback.

## Make the guide engaging and reusable

Use short sections, prediction questions with revealable answers, and a final “idea to carry forward” section. Include evidence with exact dataset, metric, setting, baseline, and version; avoid turning reported improvements into universal guarantees. Discuss what an ablation would need to establish.

Add 6–10 meaningful study cards to `content/topics/topic-N/cards.json`, preserving existing cards. Use the established four-choice schema with explanations and a valid `relatedTopicUrl`. Cards should test mechanisms and misconceptions, not title recall. Verify they also appear in printable study material.

## Integrate and finish

- Keep the existing stable paper route; update metadata to `ready` only when the guide is complete.
- Copy the course PDF into `public/papers/` and link it. Update the topic overview and README scope.
- Use site-relative Markdown links; use `withBase` in Astro for the GitHub Pages base path.
- Wire any custom component and its table-of-contents entry only to the intended paper.
- Run `npm run build`. Check rendered math, every new image, local links and heading anchors, desktop/mobile layouts, dark mode, control endpoints/midpoints, and correct/incorrect quiz feedback. Inspect screenshots visually, including detailed diagrams. Ensure no horizontal page overflow and useful figure enlargement.
- Record real verification and remaining limitations in `docs/VALIDATION.md`. Do not claim deployment unless verified; normal editing does not authorize a production push.

A finished guide must explain **what happens, where the math happens, and why the design makes sense**. Many images without these connections do not satisfy the request.
