"""Rule-based sentence and word splitter for Kazakh, following KazNERD tokenisation.

KazNERD conventions (checked in the data): every punctuation mark is its own token, including
the dot of an abbreviation and the comma inside '12,4' (-> '12', ',', '4'); words joined by a
hyphen stay one token ('бизнес-климатын', '1-інші').
"""

from __future__ import annotations

import re

# A word is letters/digits, optionally joined by hyphens; any other non-space char is a token.
WORD_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]")
HEADING_RE = re.compile(r"^=+.*=+$")  # '== Өмірбаяны ==' in Wikipedia plain text

SENTENCE_ENDS = {".", "!", "?", "…"}
CLOSERS = {"»", '"', "”", ")", "]"}
# A sentence may start with these. Not '(': after 'Алматы.” (құрастырылған ...' the text goes on.
OPENERS = {"«", '"', "“", "—", "–", "-"}
# Quotes and brackets that come in pairs; a '.' between a pair does not end the sentence
# ('“Желтоқсан. 1986. Алматы.”', '(1991, реж. Т.Теменов)').
PAIRS = {"«": "»", "“": "”", "(": ")", '"': '"'}
# Abbreviations that are often followed by a capital letter or a number. Single letters
# (initials 'А.', 'ж.' = жылы, 'ғ.' = ғасыр) are handled separately.
ABBREVIATIONS = {"жж", "ғғ", "млн", "млрд", "мыс", "проф", "акад", "доц", "құр", "бет", "реж"}


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def inside_pairs(words: list[str]) -> list[bool]:
    """inside[k] is True if words[k] lies between a matching pair of quotes or brackets.

    An opening mark without its closing mark on the same line is ignored.
    """
    inside = [False] * len(words)
    stack: list[int] = []  # positions of opening marks that are not closed yet
    for k, word in enumerate(words):
        if stack and word == PAIRS[words[stack[-1]]]:
            opener = stack.pop()
            for j in range(opener + 1, k):
                inside[j] = True
        elif word in PAIRS:
            stack.append(k)
    return inside


def _is_boundary(words: list[str], end_mark: int, last: int, inside: list[bool]) -> bool:
    """Does the sentence end at words[last]? words[end_mark] is the '.', '!', '?' or '…'."""
    if inside[last]:
        return False  # still inside quotes or brackets
    if words[end_mark] == "." and end_mark > 0:
        previous = words[end_mark - 1]
        if len(previous) == 1 and previous.isalpha():
            return False  # initial or one-letter abbreviation: 'А. Байтұрсынұлы', '1845 ж.'
        if previous.lower() in ABBREVIATIONS:
            return False
    if last + 1 >= len(words):
        return True
    following = words[last + 1]
    return following[0].isupper() or following[0].isdigit() or following in OPENERS


def split_sentences(text: str) -> list[list[str]]:
    """Split text into sentences, each a list of words. Headings and empty lines are skipped."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line or HEADING_RE.match(line):
            continue
        words = split_words(line)
        inside = inside_pairs(words)
        start = 0
        i = 0
        while i < len(words):
            if words[i] in SENTENCE_ENDS:
                last = i
                # Keep '?!', '...' and closing quotes/brackets with the sentence they end.
                while last + 1 < len(words) and (
                    words[last + 1] in SENTENCE_ENDS or words[last + 1] in CLOSERS
                ):
                    last += 1
                if _is_boundary(words, i, last, inside):
                    sentences.append(words[start : last + 1])
                    start = last + 1
                i = last + 1
            else:
                i += 1
        if start < len(words):
            sentences.append(words[start:])
    return sentences
