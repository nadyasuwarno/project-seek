"""Layer 07 Compare: per-dimension similarity, no-match, and counterexamples.

Rules, in the order the PRD asks for them:

R4  Similarity is decomposed by dimension. The headline number is only the
    average of the dimension scores, and the dimensions are always shown.
R6  If nothing in the archive clears MATCH_THRESHOLD on enough dimensions,
    the answer is "no meaningful match", with the closest candidates listed
    as weak so the researcher can see why.
R7  A counterexample is a system that looks alike on one dimension but is
    clearly different on another (>= 0.75 on one, <= 0.35 on another). The
    sharpest few are shown next to the matches to guard against pattern
    projection.

Shared absence: if neither system shows a rhythm, their periodicity
"agrees" trivially. That agreement is displayed but not scored, otherwise
every pair of non-rhythmic systems would get a free high dimension.
R8  Causal mechanism is always "unknown": similarity is not causality.

Uncertainty: each feature's bootstrap error is carried into a low/high band
around its similarity, so a comparison between two noisy signatures shows a
wide band even when the point estimate looks convincing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import FAMILIES, FEATURES, Feature, features_in

MATCH_THRESHOLD = 0.65
MIN_COMPARABLE_DIMENSIONS = 3
COUNTER_HIGH = 0.75
COUNTER_LOW = 0.35
MAX_COUNTEREXAMPLES = 3
RHYTHM_PRESENT = 0.15   # periodicity strength below this = no meaningful rhythm


def _both_below(fa: dict, fb: dict, key: str, threshold: float) -> bool:
    a, b = fa.get(key), fb.get(key)
    return a is not None and b is not None and a < threshold and b < threshold


@dataclass
class FeatureComparison:
    key: str
    label: str
    unit: str
    a: float | None
    b: float | None
    similarity: float | None
    low: float | None
    high: float | None
    used: bool          # counted in the dimension score (False = context only, or missing)
    note: str = ""


@dataclass
class DimensionComparison:
    family: str
    label: str
    score: float | None
    low: float | None
    high: float | None
    features: list[FeatureComparison]
    status: str = "scored"     # scored | shared_absence | unmeasured
    note: str = ""

    @property
    def comparable(self) -> bool:
        return self.status == "scored" and self.score is not None


@dataclass
class Comparison:
    target: dict                       # the candidate's signature record with full provenance
    dimensions: list[DimensionComparison]
    score: float | None                # mean of comparable dimension scores
    low: float | None
    high: float | None
    kind: str                          # match | weak | counterexample | insufficient
    reason: str
    scale_ratio: float | None
    causal_status: str = "unknown"

    @property
    def n_comparable(self) -> int:
        return sum(d.comparable for d in self.dimensions)

    def dimension(self, family: str) -> DimensionComparison:
        return next(d for d in self.dimensions if d.family == family)

    def to_dict(self) -> dict:
        return {
            "target_observation_id": self.target.get("observation_id"),
            "target_system": self.target.get("system_name"),
            "score": self.score, "low": self.low, "high": self.high, "kind": self.kind,
            "reason": self.reason, "scale_ratio": self.scale_ratio, "causal_status": self.causal_status,
            "dimensions": {d.family: d.score for d in self.dimensions},
        }


@dataclass
class SearchResult:
    query: dict
    matches: list[Comparison] = field(default_factory=list)
    weak: list[Comparison] = field(default_factory=list)
    counterexamples: list[Comparison] = field(default_factory=list)
    insufficient: list[Comparison] = field(default_factory=list)
    scale_free: bool = True
    discovery_prompt: str | None = None

    @property
    def no_match(self) -> bool:
        return not self.matches


def _transform(f: Feature, v: float) -> float | None:
    if f.transform == "log":
        return math.log10(v) if v > 0 else None
    return v


def _transformed_error(f: Feature, v: float, err: float | None) -> float:
    if not err:
        return 0.0
    if f.transform == "log":
        return err / (v * math.log(10)) if v > 0 else 0.0
    return err


def compare_feature(f: Feature, a: float | None, b: float | None, ea: float | None, eb: float | None,
                    scale_free: bool) -> FeatureComparison:
    use = f.scale_free or not scale_free
    if a is None or b is None:
        missing = "not measured in " + ("either" if a is None and b is None else "the query" if a is None else "the candidate")
        return FeatureComparison(f.key, f.label, f.unit, a, b, None, None, None, False, missing)
    ta, tb = _transform(f, a), _transform(f, b)
    if ta is None or tb is None:
        return FeatureComparison(f.key, f.label, f.unit, a, b, None, None, None, False, "non-positive value")
    diff = abs(ta - tb)
    err = math.hypot(_transformed_error(f, a, ea), _transformed_error(f, b, eb))
    sim = math.exp(-diff / f.tolerance)
    low = math.exp(-(diff + err) / f.tolerance)
    high = math.exp(-max(0.0, diff - err) / f.tolerance)
    note = "" if use else "absolute scale: shown as context, not scored"
    return FeatureComparison(f.key, f.label, f.unit, a, b, sim, low, high, use, note)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare(query: dict, target: dict, scale_free: bool = True) -> Comparison:
    fa, fb = query["features"], target["features"]
    ua, ub = query.get("uncertainty", {}), target.get("uncertainty", {})

    dimensions = []
    for family, meta in FAMILIES.items():
        comps = [compare_feature(f, fa.get(f.key), fb.get(f.key), ua.get(f.key), ub.get(f.key), scale_free)
                 for f in features_in(family)]
        scored = [c for c in comps if c.used]
        dim = DimensionComparison(
            family, meta["label"],
            _mean([c.similarity for c in scored]),
            _mean([c.low for c in scored]),
            _mean([c.high for c in scored]),
            comps,
        )
        if dim.score is None:
            dim.status, dim.note = "unmeasured", "Not measured in one or both signatures."
        elif family == "temporal" and _both_below(fa, fb, "periodicity_strength", RHYTHM_PRESENT):
            dim.status = "shared_absence"
            dim.note = "Neither shows a rhythm. Agreeing on an absence is not evidence of similarity, so it is not scored."
        dimensions.append(dim)

    comparable = [d for d in dimensions if d.comparable]
    score = _mean([d.score for d in comparable])
    low = _mean([d.low for d in comparable])
    high = _mean([d.high for d in comparable])

    scale_ratio = None
    if fa.get("mean_speed") and fb.get("mean_speed"):
        scale_ratio = max(fa["mean_speed"], fb["mean_speed"]) / min(fa["mean_speed"], fb["mean_speed"])

    kind, reason = _classify(comparable, score)
    return Comparison(target, dimensions, score, low, high, kind, reason, scale_ratio)


def _classify(comparable: list[DimensionComparison], score: float | None) -> tuple[str, str]:
    if len(comparable) < MIN_COMPARABLE_DIMENSIONS:
        return "insufficient", (f"Only {len(comparable)} dimension(s) could be measured in both; "
                                f"at least {MIN_COMPARABLE_DIMENSIONS} are needed to rank.")
    best = max(comparable, key=lambda d: d.score)
    worst = min(comparable, key=lambda d: d.score)
    if score is not None and score >= MATCH_THRESHOLD and worst.score > COUNTER_LOW:
        return "match", (f"Strongest agreement on {best.label.lower()} ({best.score:.2f}); "
                         f"weakest on {worst.label.lower()} ({worst.score:.2f}).")
    if best.score >= COUNTER_HIGH and worst.score <= COUNTER_LOW:
        return "counterexample", (f"Similar {best.label.lower()} ({best.score:.2f}) but clearly different "
                                  f"{worst.label.lower()} ({worst.score:.2f}).")
    return "weak", f"Average similarity {score:.2f} is below the {MATCH_THRESHOLD:.2f} match threshold."


def _contrast(c: Comparison) -> float:
    scores = [d.score for d in c.dimensions if d.comparable]
    return max(scores) - min(scores) if scores else 0.0


def search(query: dict, candidates: list[dict], scale_free: bool = True) -> SearchResult:
    """Compare a query signature against every archive signature except itself."""
    result = SearchResult(query=query, scale_free=scale_free)
    for cand in candidates:
        if cand.get("observation_id") == query.get("observation_id"):
            continue
        c = compare(query, cand, scale_free)
        {"match": result.matches, "weak": result.weak, "counterexample": result.counterexamples,
         "insufficient": result.insufficient}[c.kind].append(c)
    # Keep only the sharpest counterexamples; the rest are simply weak candidates.
    result.counterexamples.sort(key=_contrast, reverse=True)
    for c in result.counterexamples[MAX_COUNTEREXAMPLES:]:
        c.kind = "weak"
        result.weak.append(c)
    result.counterexamples = result.counterexamples[:MAX_COUNTEREXAMPLES]
    for bucket in (result.matches, result.weak):
        bucket.sort(key=lambda c: c.score or 0, reverse=True)
    result.discovery_prompt = discovery_prompt(query, result)
    return result


def _feature_phrase(dim: DimensionComparison) -> str:
    used = sorted((f for f in dim.features if f.used and f.similarity is not None),
                  key=lambda f: f.similarity, reverse=True)
    if not used:
        return ""
    f = used[0]
    return f" ({f.label.lower()} {f.a:.2f} vs {f.b:.2f})"


def discovery_prompt(query: dict, result: SearchResult) -> str | None:
    """A question to investigate, never a conclusion (PRD: discovery prompt, labeled exploratory)."""
    if result.matches:
        top = result.matches[0]
        comparable = [d for d in top.dimensions if d.comparable]
        best = max(comparable, key=lambda d: d.score)
        worst = min(comparable, key=lambda d: d.score)
        scale = ""
        if top.scale_ratio and top.scale_ratio >= 10:
            scale = f", even though their absolute speeds differ about {top.scale_ratio:.0f}x"
        return (f"{query.get('system_name')} and {top.target.get('system_name')} share a similar "
                f"{best.label.lower()}{_feature_phrase(best)}{scale}. "
                f"What would have to be true for both to be produced by a comparable mechanism, "
                f"and does their weakest agreement, on {worst.label.lower()} ({worst.score:.2f}), rule that out?")
    if result.counterexamples:
        c = result.counterexamples[0]
        return (f"No strong match. {c.target.get('system_name')} is a useful counterexample: {c.reason} "
                f"Which of the dimensions actually defines the behavior you are looking at?")
    return None
