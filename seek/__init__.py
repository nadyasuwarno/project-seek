"""Seeker: search observed behavior across systems.

Pipeline (PRD architecture layers):
  01 Acquire      -> storage.py   (video file + contributor + metadata)
  02 Normalize    -> extraction.py (pixels -> micrometres, frames -> seconds)
  03 Identify     -> db.py        (system, domain, conditions, provenance)
  04 Measure      -> extraction.py (trackpy trajectories)
  05 Represent    -> features.py  (behavioral signature, schema.py)
  06 Index        -> db.py        (signature archive)
  07 Compare      -> compare.py   (per-dimension similarity, no-match, counterexamples)
  08-09 Explore / Interpret -> app.py (Seeker Terminal UI)
"""

__version__ = "0.1.0"
