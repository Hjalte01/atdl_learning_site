# Paper learning-site authoring

## Browser verification on terminal-only hosts

Use the pinned development shell with `nix develop`. It supplies Node, PDF tools,
Python Playwright and NixOS-compatible Chromium; no desktop/display is required.
After `npm run build`, start `npm run preview -- --host 127.0.0.1 --port 4322`.
Run `nix develop -c python scripts/browser-check.py http://127.0.0.1:4322/atdl_learning_site/papers/<slug>`
from this repository. Review the screenshots in `.validation/browser/` with the
image viewer, including enlarged figures and mobile equations. Automated checks
alone are not a visual review. Extend checks for a guide's custom interactive lab
when applicable. Do not borrow Playwright from another repository, download
generic Linux browser binaries, or put the only verification scripts in `/tmp`.
Stop the preview with `npm run preview -- stop` when finished.

For requests to create or expand paper guides (including “create sites for topic 2-4”), read and follow [GPT.md](GPT.md). Topic-paper shorthand means one paper unless the user explicitly requests a range. Preserve existing work and verify against the supplied course paper.

## Automatic next-guide requests

“Goal create site”, “create a site”, or “create the next missing site” without a specific target means complete **one** unfinished learning guide. Read [GPT.md](GPT.md), especially automatic selection. Run `python3 scripts/materials.py next` to refresh from the live study vault and inspect the next candidate. This command syncs automatically; do not decide that work is finished from the saved catalog alone. A nonzero exit or `blocked` result means source access must be resolved, not that all guides are complete. `queued` paper metadata is the same as the UI label **Not started**. Never interpret an unspecified request as permission to generate every guide.

Priority: unfinished papers first, highest available topic number first (Topic 5 before 4, 3, 2, 1; Topic 6 first when supplied), then ascending paper number within that topic. After all papers, do overviews, then slides, then extra material; within each category use descending topic and ascending reading order. Explicit user targets override this default. Topics without supplied readings are not candidates. If the selected source is unavailable, report the missing source rather than inventing content or silently choosing a lower-priority guide.

The PDF inventory belongs in `content/materials/catalog.json`. The live source is configured in `content/materials/source.json` (currently `/srv/obsidian-webdav/study/ATDL`); `ATDL_VAULT` or `--vault` can override it. Refresh without selecting with `python3 scripts/materials.py sync`, or import another vault with `python3 scripts/materials.py import /path/to/study/ATDL`; review inferred classifications against the vault. `next --offline` is only for explicit cached inspection and never proves that the current vault is complete. Builds use committed PDFs and do not require vault access. Preserve existing guides, source PDFs, and metadata. Never claim the whole vault is included until its inventory has actually been imported and verified.
