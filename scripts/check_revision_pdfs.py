"""Check compiled manuscript content, text colors, bounds, and LaTeX warnings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def equivalent_text(text):
    """Ignore word hyphenation while preserving numeric signs and x-y algebra."""
    text = " ".join(text.split())
    return re.sub(r"(?<=[A-Za-z]{2})-\s*(?=[A-Za-z]{2})", "", text)


def inspect(path):
    import pdfplumber

    pages, colors, parts, outside = [], {}, [], 0
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, 1):
            lines = (page.extract_text() or "").splitlines()
            if lines and lines[-1].strip().isdigit():
                lines = lines[:-1]
            parts.append(" ".join(lines))
            page_colors = {}
            for char in page.chars:
                color = str(char.get("non_stroking_color"))
                colors[color] = colors.get(color, 0) + 1
                page_colors[color] = page_colors.get(color, 0) + 1
                outside += int(
                    char["x0"] < -1
                    or char["x1"] > page.width + 1
                    or char["top"] < -1
                    or char["bottom"] > page.height + 1
                )
            pages.append(
                dict(page=index, width=page.width, height=page.height, characters=len(page.chars), colors=page_colors)
            )
    return dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        pages=pages,
        color_counts=colors,
        characters_outside_page=outside,
    ), " ".join(" ".join(parts).split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    args = parser.parse_args()
    results, texts = {}, {}
    for name in ("main", "main_marked", "response_to_reviewers"):
        results[name], texts[name] = inspect(args.paper / f"{name}.pdf")
        log = (args.paper / f"{name}.log").read_text(errors="replace")
        results[name]["overfull_boxes"] = len(re.findall(r"Overfull \\[hv]box", log))
        results[name]["undefined_references_or_citations"] = bool(
            re.search(r"(?:Citation|Reference).+undefined|There were undefined", log)
        )
        if (
            results[name]["characters_outside_page"]
            or results[name]["overfull_boxes"]
            or results[name]["undefined_references_or_citations"]
        ):
            raise ValueError(f"PDF structural validation failed: {name}")
    results["clean_marked_content_identical_ignoring_page_numbers_and_whitespace"] = (
        texts["main"] == texts["main_marked"]
    )
    results["clean_marked_content_equivalent_after_word_hyphenation_normalization"] = equivalent_text(
        texts["main"]
    ) == equivalent_text(texts["main_marked"])
    if not results["clean_marked_content_equivalent_after_word_hyphenation_normalization"]:
        raise ValueError("Clean and marked manuscript content differs")
    if not any("0.275" in color or "0.27452" in color for color in results["main_marked"]["color_counts"]):
        raise ValueError("Marked PDF has no requested blue text")
    first_colors = results["main_marked"]["pages"][0]["colors"]
    if sum(count for color, count in first_colors.items() if "0.275" in color or "0.27452" in color) < 1000:
        raise ValueError("The revised abstract was not actually colored inside its stored front-matter box")
    if results["main_marked"]["pages"][2]["colors"].get("(0.0,)", 0) < 1000:
        raise ValueError("Abstract color leaked into the unchanged introduction")
    results["marking"] = json.loads((args.paper / "revision_evidence/marked_changes.json").read_text())
    target = args.paper / "validation_20261021/pdf_structure_checks.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: dict(
                    pages=len(results[name]["pages"]),
                    overfull=results[name]["overfull_boxes"],
                    outside_page=results[name]["characters_outside_page"],
                )
                for name in ("main", "main_marked", "response_to_reviewers")
            },
            indent=2,
        )
    )
    print("Clean/marked text matches after word-hyphenation normalization; requested blue is present.")


if __name__ == "__main__":
    main()
