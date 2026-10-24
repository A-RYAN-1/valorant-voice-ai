"""Pure text helpers shared by the data steps."""

import re

QUOTES = "\"'‘’“”"


def clean_text(raw: str) -> str:
    """Trims whitespace and the quote marks the source transcripts wrap every line in."""
    text = " ".join(raw.split())
    if len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES:
        text = text[1:-1].strip()
    return text


def text_key(text: str) -> str:
    """Identity of a line for de-duplication and splitting: words only, lower case."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
