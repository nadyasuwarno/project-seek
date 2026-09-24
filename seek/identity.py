"""Contributor identity: every upload is tied to a researcher's ORCID iD.

ORCID iDs are 16 characters (four groups of four) where the last character is
an ISO 7064 Mod 11-2 checksum. Validating the checksum catches typos before a
contribution gets attributed to the wrong person.
"""

from __future__ import annotations

import re

_ORCID_RE = re.compile(r"^(\d{4})-?(\d{4})-?(\d{4})-?(\d{3}[\dX])$")


def orcid_checksum(base_digits: str) -> str:
    """Return the check character for the first 15 digits of an ORCID iD."""
    total = 0
    for ch in base_digits:
        total = (total + int(ch)) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)


def normalize_orcid(value: str) -> str:
    """Accept '0000-0002-1825-0097', '0000000218250097' or an orcid.org URL.

    Returns the canonical hyphenated form, or raises ValueError with a reason.
    """
    if value is None:
        raise ValueError("ORCID iD is required.")
    raw = value.strip().upper()
    raw = re.sub(r"^HTTPS?://(WWW\.)?ORCID\.ORG/", "", raw)
    m = _ORCID_RE.match(raw)
    if not m:
        raise ValueError("ORCID iD should look like 0000-0002-1825-0097.")
    digits = "".join(m.groups())
    if orcid_checksum(digits[:15]) != digits[15]:
        raise ValueError("That ORCID iD fails its checksum; check for a typo.")
    return "-".join(m.groups())


def orcid_url(orcid: str | None) -> str | None:
    return f"https://orcid.org/{orcid}" if orcid else None
