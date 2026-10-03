"""Check one completed guide against a running preview; save review captures."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url", help="Full guide URL, including the site's base path")
parser.add_argument("--output", default=".validation/browser")
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
slug = urlsplit(args.url).path.rstrip("/").split("/")[-1]
assert (Path("content/papers") / slug / "index.md").is_file(), "Run from the repository root"
base = args.url.split("/papers/", 1)[0]
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=os.environ["ATDL_CHROMIUM"])
    context = browser.new_context()
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    assert page.goto(args.url).status == 200
    page.locator("article.prose img").evaluate_all("imgs => Promise.all(imgs.map(i => {i.loading='eager'; return i.decode()}))")
    images = page.locator("article.prose img")
    assert images.count() > 0, "Missing teaching figures"
    assert page.locator(".katex").count() > 0, "Missing rendered math"
    assert page.locator(".katex-error").count() == 0
    for href in set(page.locator("a[href]").evaluate_all("as => as.map(a => a.getAttribute('href'))")):
        target = urljoin(args.url, href)
        if urlsplit(target).netloc != urlsplit(args.url).netloc:
            continue
        response = context.request.get(target)
        assert response.status == 200, f"Broken link: {target} ({response.status})"
        fragment = unquote(urlsplit(target).fragment)
        if fragment:
            if urlsplit(target).path.rstrip("/") == urlsplit(args.url).path.rstrip("/"):
                assert page.evaluate("id => Boolean(document.getElementById(id))", fragment), target
            elif "text/html" in response.headers.get("content-type", ""):
                linked = context.new_page()
                linked.goto(target)
                assert linked.evaluate("id => Boolean(document.getElementById(id))", fragment), target
                linked.close()
    for width in (1440, 390):
        for theme in ("light", "dark"):
            page.set_viewport_size({"width": width, "height": 844 if width == 390 else 1000})
            page.evaluate("t => {document.documentElement.dataset.theme=t; window.scrollTo({top:0,behavior:'instant'})}", theme)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow: {width} {theme}"
            prefix = output / f"{slug}-{width}-{theme}"
            page.screenshot(path=f"{prefix}-top.png")
            page.screenshot(path=f"{prefix}-full.png", full_page=True)
            for index in range(page.locator(".katex-display").count()):
                page.locator(".katex-display").nth(index).scroll_into_view_if_needed()
                page.screenshot(path=f"{prefix}-math-{index}.png")
    page.set_viewport_size({"width": 1440, "height": 1000})
    cards = page.locator("[data-study-card]")
    ids = cards.evaluate_all("cs => cs.map(c => c.dataset.cardId)")
    assert ids, "Missing study cards"
    questions = []
    for index, card_id in enumerate(ids):
        card = page.locator(f'[data-card-id="{card_id}"]')
        questions.append(card.locator(".study-question").inner_text())
        correct = int(card.locator("form").get_attribute("data-correct-index"))
        card.locator(f'input[value="{correct}"]').check()
        card.locator('button[type="submit"]').click()
        assert card.locator("[data-feedback-status]").inner_text() == "Correct."
        if index < len(ids) - 1:
            page.locator("[data-next-card]").click()
    page.reload()
    first = page.locator(f'[data-card-id="{ids[0]}"]')
    correct = int(first.locator("form").get_attribute("data-correct-index"))
    first.locator(f'input[value="{(correct + 1) % 4}"]').check()
    first.locator('button[type="submit"]').click()
    assert "Not quite" in first.locator("[data-feedback-status]").inner_text()
    page.goto(f"{base}/study/print")
    for question in questions:
        assert question in page.inner_text("body"), f"Missing printable card: {question}"
    diagram_count = 0
    for asset in sorted((Path("public/figures") / slug).glob("*.svg")):
        page.goto(f"{base}/figures/{slug}/{asset.name}")
        page.locator("svg").screenshot(path=str(output / f"{slug}-detail-{asset.stem}.png"))
        bounds = page.evaluate("""() => {
          const svg=document.querySelector('svg'), v=svg.viewBox.baseVal;
          const boxes=[...svg.querySelectorAll('rect[rx]')].map(r=>r.getBBox());
          return [...svg.querySelectorAll('text')].flatMap(t=>{
            const b=t.getBBox(); const errors=[];
            if(b.x<v.x || b.y<v.y || b.x+b.width>v.x+v.width || b.y+b.height>v.y+v.height) errors.push(t.textContent+' outside SVG');
            const box=boxes.find(r=>b.x>=r.x && b.x<r.x+r.width && b.y>=r.y && b.y<r.y+r.height);
            if(box && b.x+b.width>box.x+box.width-5) errors.push(t.textContent+' outside panel');
            return errors;
          });
        }""")
        assert not bounds, f"{asset}: {bounds}"
        diagram_count += 1
    assert not errors, errors
    browser.close()
result = {"url": args.url, "cards": len(ids), "diagrams": diagram_count,
          "viewports": [1440, 390], "themes": ["light", "dark"], "status": "passed"}
(output / f"{slug}-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
print(f"Visually inspect the screenshots in {output}; automated checks do not establish visual quality.")
