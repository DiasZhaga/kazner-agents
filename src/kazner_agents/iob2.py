"""Pure functions on word-level IOB2 labels: spans -> labels, repair, validation, labels -> spans.

IOB2: the first word of an entity gets B-TYPE, the following words of the same entity I-TYPE,
all other words O. An I-TYPE label is only valid right after B-TYPE or I-TYPE of the same type.
"""

from __future__ import annotations

from kazner_agents.labels import IOB2_LABELS
from kazner_agents.messages import Span


def _squash(text: str) -> str:
    """Lower-case and remove all whitespace, so that 'Абай  Құнанбайұлы' == 'абайқұнанбайұлы'."""
    return "".join(text.split()).lower()


# LLMs often write the dictionary form of the last word ('Қазақстан' for 'Қазақстанның').
# A text that matches the words except for such an ending is accepted. Kazakh suffix chains
# are rarely longer than 12 letters; very short texts (under 3 letters) must match exactly.
MAX_SUFFIX = 12


def _same_or_suffixed(joined: str, target: str) -> bool:
    if joined == target:
        return True
    return (
        len(target) >= 3
        and joined.startswith(target)
        and len(joined) - len(target) <= MAX_SUFFIX
    )


def text_matches(words: list[str], start: int, end: int, text: str) -> bool:
    """Do words[start..end] spell `text` (allowing a suffix on the last word)?"""
    return _same_or_suffixed(_squash("".join(words[start : end + 1])), _squash(text))


def locate_text(words: list[str], text: str, near: int) -> tuple[int, int] | None:
    """Find the words that spell `text`; prefer the match closest to `near`.

    Exact matches win over matches that only differ by a suffix on the last word.
    Returns (start, end) word indices, inclusive, or None if the text is not in the sentence.
    """
    target = _squash(text)
    if not target:
        return None
    exact, suffixed = [], []
    for start in range(len(words)):
        joined = ""
        for end in range(start, len(words)):
            joined += _squash(words[end])
            if joined == target:
                exact.append((start, end))
            elif _same_or_suffixed(joined, target):
                suffixed.append((start, end))
            if len(joined) >= len(target):
                break
    matches = exact or suffixed
    if not matches:
        return None
    return min(matches, key=lambda match: abs(match[0] - near))


def spans_to_iob2(words: list[str], spans: list[Span]) -> tuple[list[str], list[str]]:
    """Turn entity spans into one IOB2 label per word.

    Invalid spans are repaired or dropped, and every change is described in `repairs`:
    start after end (swapped), text that does not match the words (moved or dropped),
    span outside the sentence (cut or dropped), overlap with an earlier span (dropped).
    """
    n = len(words)
    labels = ["O"] * n
    repairs: list[str] = []
    # Earlier spans first; for the same start the longer span first, so it wins an overlap.
    ordered = sorted(
        spans, key=lambda s: (min(s.start_word, s.end_word), -abs(s.end_word - s.start_word))
    )
    for span in ordered:
        start, end, etype = span.start_word, span.end_word, span.type
        name = f"{etype} span {start}-{end}"
        if start > end:
            start, end = end, start
            repairs.append(f"swapped start and end of {name}")
        if span.text and not text_matches(words, start, end, span.text):
            found = locate_text(words, span.text, near=start)
            if found is None:
                repairs.append(f"dropped {name}: its text {span.text!r} is not in the sentence")
                continue
            repairs.append(f"moved {name} to {found[0]}-{found[1]} to match its text")
            start, end = found
        if start >= n:
            repairs.append(f"dropped {name}: outside the sentence of {n} words")
            continue
        if end >= n:
            repairs.append(f"cut {name} at the last word {n - 1}")
            end = n - 1
        if any(label != "O" for label in labels[start : end + 1]):
            repairs.append(f"dropped {name}: overlaps another entity")
            continue
        labels[start] = f"B-{etype}"
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{etype}"
    return labels, repairs


def repair_iob2(labels: list[str]) -> tuple[list[str], list[str]]:
    """Fix an invalid IOB2 sequence: unknown labels become O, a stray I-X becomes B-X."""
    fixed: list[str] = []
    repairs: list[str] = []
    previous = "O"
    for i, label in enumerate(labels):
        if label not in IOB2_LABELS:
            repairs.append(f"word {i}: unknown label {label!r} replaced by O")
            label = "O"
        elif label.startswith("I-") and previous[2:] != label[2:]:
            new_label = "B-" + label[2:]
            repairs.append(f"word {i}: {label} does not continue an entity, changed to {new_label}")
            label = new_label
        fixed.append(label)
        previous = label
    return fixed, repairs


def validate_iob2(labels: list[str]) -> list[str]:
    """Return a list of problems; an empty list means the sequence is valid IOB2."""
    problems = []
    previous = "O"
    for i, label in enumerate(labels):
        if label not in IOB2_LABELS:
            problems.append(f"word {i}: unknown label {label!r}")
        elif label.startswith("I-") and previous[2:] != label[2:]:
            problems.append(f"word {i}: {label} after {previous}")
        previous = label
    return problems


def iob2_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """Read entities from a valid IOB2 sequence as (start, end, type), end inclusive."""
    spans = []
    start, etype = None, None
    for i, label in enumerate(labels + ["O"]):  # the extra O closes an entity at the end
        if label.startswith("I-") and etype == label[2:]:
            continue
        if start is not None:
            spans.append((start, i - 1, etype))
            start, etype = None, None
        if label.startswith("B-"):
            start, etype = i, label[2:]
    return spans
