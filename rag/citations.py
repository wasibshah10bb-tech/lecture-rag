"""Milestone 11: check the citations in an LLM answer.

The model is told to copy each excerpt's label, e.g. [tcs-04-parsing.pdf, p. 32].
It usually does, but not always (we saw "[tc-04-parsing.pdf, p. 32]" and
"[notes.txt]"). This module compares every citation in the answer with the
labels of the chunks the model was actually given.

What it can check: does each citation point to an excerpt the model saw?
What it cannot check: does that excerpt actually support the claim?
"""

import difflib
import re

# Any [...] that mentions a supported file counts as a citation attempt, so
# malformed variants like "[file.pdf, p. 8, 9]" get flagged instead of ignored.
ATTEMPT_RE = re.compile(r"\[([^\[\]]*?\.(?:pdf|md|txt)\b[^\[\]]*)\]", re.IGNORECASE)
# A well-formed label: "<file>, p. <n>" or "<file>, line <n>".
PARTS_RE = re.compile(r"^(.+?\.(?:pdf|md|txt)), (p\. \d+|line \d+)$", re.IGNORECASE)

# How similar a cited file name must be to a given one to count as a typo (0..1).
FILENAME_SIMILARITY = 0.85


def citation(hit):
    """The label a chunk gets in the prompt, which is also the citation format:
    [file.pdf, p. 3] for PDFs, [notes.md, line 42] for Markdown/TXT."""
    if hit.get("page") is not None:
        return f"[{hit['source']}, p. {hit['page']}]"
    return f"[{hit['source']}, line {hit['line']}]"


def _normalize(label):
    """Tidy spacing so 'tcs.pdf,p.32' and 'tcs.pdf, p. 32' compare equal."""
    label = " ".join(label.split())
    label = re.sub(r"\s*,\s*", ", ", label)
    return re.sub(r"\bp\.\s*(\d)", r"p. \1", label)


def check(reply, hits):
    """Validate and, where safe, repair the citations in reply.

    Returns (checked_reply, report) where report is
      {'valid':    [label, ...],            # cited exactly as given
       'repaired': [(wrong, right), ...],   # file-name typo, fixed in checked_reply
       'invalid':  [label, ...],            # not among the given excerpts, left as is
       'uncited':  bool}                    # the answer contains no citation at all

    A repair happens only if the page/line number matches exactly and exactly one
    given label has a file name that differs by a small typo. A wrong number is
    never repaired: we can't know which page the model meant.
    """
    given = [citation(h)[1:-1] for h in hits]  # labels without the brackets
    report = {"valid": [], "repaired": [], "invalid": [], "uncited": False}

    def fix(match):
        label = _normalize(match.group(1))
        if label in given:
            report["valid"].append(label)
            return f"[{label}]"

        parts = PARTS_RE.match(label)
        if parts:
            cited_file, location = parts.groups()
            candidates = {
                g for g in given
                if g.endswith(", " + location)
                and difflib.SequenceMatcher(None, cited_file, g.rsplit(", ", 1)[0]).ratio()
                >= FILENAME_SIMILARITY
            }
            if len(candidates) == 1:
                right = candidates.pop()
                report["repaired"].append((label, right))
                return f"[{right}]"

        report["invalid"].append(label)
        return match.group(0)  # leave it visible as the model wrote it

    checked = ATTEMPT_RE.sub(fix, reply)
    report["uncited"] = not (report["valid"] or report["repaired"] or report["invalid"])
    return checked, report


def problems(report):
    """Human-readable notes about a report; empty if all citations are fine."""
    notes = [f"repaired citation [{wrong}] -> [{right}] (file-name typo)"
             for wrong, right in report["repaired"]]
    notes += [f"citation [{label}] does not match any excerpt the model was given"
              for label in report["invalid"]]
    if report["uncited"]:
        notes.append("the answer contains no citations")
    return notes
