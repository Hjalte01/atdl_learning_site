# ATDL Learning

A static Astro learning site modeled on `study/thesis/learning_site`.

## Location

The project now lives at `/home/hjalte/Documents/study/ATDL/learning_site/`, as requested. It is inside the study Obsidian vault. No Remotely Save settings were changed; this folder is subject to that vault's existing sync configuration.

## Run

```sh
cd '/home/hjalte/Documents/study/ATDL/learning_site'
npm ci
npm run dev -- --host 127.0.0.1 --port 4322
```

Open http://localhost:4322. For production verification, run `npm run build`; `npm run preview -- --host 127.0.0.1 --port 4322` serves the static build. Node 22.12 or newer is required by this Astro version.

## Scope

Six topic routes; the twelve available papers for Topics 1 and 2 are catalogued. FLUX.1 Kontext and DiffAtlas have full visual learning guides, equations, interactive illustrations, and recall cards. DiffAtlas includes seven original diagrams and eight multiple-choice cards. Topics 3–6 await their actual titles and reading lists. Topic 1's grouping label is provisional. Other papers are explicitly queued.

The Flux explanation uses the supplied arXiv v2 PDF (24 June 2025), copied into `public/papers/flux-kontext.pdf` for local reading. Toy examples are labeled. Results are historical claims from the report, not current product comparisons.

## Add learning material

Follow [GPT.md](GPT.md) for architecture-to-math visual authoring. “Create sites for topic 2-4” selects Topic 2, paper 4.

- Topic metadata and overview: `content/topics/topic-N/index.md`.
- Paper metadata and explanation: `content/papers/<stable-slug>/index.md`.
- Cards: `content/topics/topic-N/cards.json`; `relatedTopicUrl` points to the relevant paper section.
- Shared presentation: `src/`; validation: `src/content.config.ts`.

Add a paper directory with frontmatter `title`, `shortTitle`, `topic`, `order`, `description`, and `status` (`queued` or `ready`); optionally `source` and `pdf`. It is automatically listed in its topic and receives a stable `/papers/<slug>` route. No component changes are needed for ordinary papers. Flux and DiffAtlas include paper-specific interactive components.

Use level-two headings; explain intuition before equations; define symbols; cite paper sections; distinguish reported evidence from critique. Equations use remark-math and KaTeX, with fonts hosted locally. Sources and optional visual collections inherited from the thesis layout remain available.

`/study` provides practice and `/study/print` provides printable cards. Answers are temporary browser state. The site contains no generation API, account system, or paid service dependency. Theme preference uses local browser storage.

## Checks

Run `npm run build` after changes. Check topic-to-paper links, mathematical rendering, slider endpoints, quiz feedback, and mobile layout when editing application behavior.

## GitHub Pages

Live site: https://hjalte01.github.io/atdl_learning_site/

Pushes to `main` build and deploy through `.github/workflows/deploy.yml`.
The workflow runs `npm ci` and `npm run build` before publishing `dist/`.
Astro's `base` is `/atdl_learning_site`; use `withBase` from `src/lib/url.ts`
for site-local links and assets. Markdown URLs receive the base automatically.
