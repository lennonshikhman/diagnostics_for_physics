"""Mark changed paragraphs/environments blue against the archived reviewed text.

Section sources are expanded; generated inputs retain their original layout.
Highlighting is deliberately at paragraph/environment granularity. Author and
affiliation commands are preserved. No deletions are inserted into the revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalize(value):
    return " ".join(value.split())


def expand_sections(text, root):
    pattern = re.compile(r"\\input\{(sections/[^}]+)\}")

    def expand(match):
        path = root / match.group(1)
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        if not path.is_file():
            raise FileNotFoundError(path)
        return expand_sections(path.read_text(), root)

    return pattern.sub(expand, text)


def paragraphs(text):
    current, stack = [], []
    for line in text.splitlines(keepends=True):
        current.append(line)
        for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", line):
            if environment in {"document", "frontmatter"}:
                continue
            if action == "begin":
                stack.append(environment)
            elif stack and stack[-1] == environment:
                stack.pop()
        if not line.strip() and not stack:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    parser.add_argument("--baseline", type=Path, default=Path("PAPER/originals/before_0.2_revision/main.tex"))
    args = parser.parse_args()
    canonical = (args.paper / "main.tex").read_text()
    reviewed = normalize(args.baseline.read_text())
    expanded = expand_sections(canonical, args.paper)
    preamble, body = expanded.split(r"\begin{document}", 1)
    preamble += "\n" + r"\definecolor{RevisionBlue}{RGB}{0,70,170}" + "\n"
    blocks, report = [], []
    generated_marked = args.paper / "generated_marked"
    generated_marked.mkdir(exist_ok=True)
    protected = re.compile(r"\\(?:title|author|affiliation|ead|cortext|journal|begin\{frontmatter|end\{frontmatter)")
    for block in paragraphs(body):
        # All blocks before the first main section belong to the front matter
        # or Program Summary; protect identity commands explicitly.
        identity = bool(protected.search(block)) and r"\begin{abstract}" not in block
        stripped = normalize(block)
        structural_only = stripped in {r"\end{document}", r"\appendix", ""}
        changed = bool(stripped and stripped not in reviewed and not identity and not structural_only)
        if r"\bibliography{" in block:
            changed = True
        if changed:
            # Floats reset their color, and front-matter boxes are stored for
            # later output. Set the color inside those environments as well.
            highlighted = re.sub(
                r"(\\begin\{(?:table|figure)\}[^\n]*\n)",
                lambda match: match.group(1) + "\\color{RevisionBlue}\n",
                block.rstrip() + "\n",
            )

            def marked_input(match):
                relative = Path(match.group(1))
                if relative.suffix != ".tex":
                    relative = relative.with_suffix(".tex")
                content = (args.paper / relative).read_text()
                content = re.sub(
                    r"(\\begin\{(?:table|figure)\}[^\n]*\n)",
                    lambda begin: begin.group(1) + "\\color{RevisionBlue}\n",
                    content,
                )
                target = generated_marked / relative.name
                target.write_text(content)
                return r"\input{generated_marked/" + relative.name + "}"

            highlighted = re.sub(r"\\input\{(generated/[^}]+)\}", marked_input, highlighted)
            if r"\begin{abstract}" in highlighted:
                # Close color inside elsarticle's stored vbox, before unboxing.
                highlighted = highlighted.replace(
                    r"\begin{abstract}", "\\begin{abstract}\n\\begingroup\\color{RevisionBlue}"
                ).replace(r"\end{abstract}", "\\par\\endgroup\n\\end{abstract}")
                blocks.append("\n" + highlighted + "\n\n")
            else:
                blocks.append("\n" + r"\begingroup\color{RevisionBlue}" + "\n" + highlighted + r"\endgroup" + "\n\n")
        else:
            blocks.append(block)
        report.append(
            dict(sha256=hashlib.sha256(block.encode()).hexdigest(), highlighted=changed, protected_identity=identity)
        )
    marked = preamble + r"\begin{document}" + "".join(blocks)
    (args.paper / "main_marked.tex").write_text(marked)
    metadata = dict(
        colour="blue",
        rgb=[0, 70, 170],
        baseline=str(args.baseline),
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        canonical_sha256=hashlib.sha256((args.paper / "main.tex").read_bytes()).hexdigest(),
        method="Changed complete paragraphs/environments in blue; section inputs expanded; generated inputs copied with internal float color commands; internal abstract/float commands preserve blue in stored boxes; updated bibliography marked as one block.",
        changed_blocks=sum(x["highlighted"] for x in report),
        total_blocks=len(report),
        blocks=report,
    )
    path = args.paper / "revision_evidence/marked_changes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Marked {metadata['changed_blocks']} of {len(report)} blocks blue (RGB 0,70,170).")


if __name__ == "__main__":
    main()
