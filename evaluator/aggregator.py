"""Cross-scenario aggregation for evaluation runs.

Takes the **list of per-conversation detail dicts** written by
``runner/cli.py`` (each one carrying ``agent_name``, ``scenario``,
``report.layered_report``, ``conversation``…) and produces the four
artefacts the static HTML report and the dashboard need:

  * ``matrix``        — per-agent × per-instruction × per-profile mean score,
                        with the contributing ``scenario_ids`` preserved so
                        multiple personas sharing the same ``profile_id``
                        don't silently overwrite each other.
  * ``radar``         — per-agent mean of each L3 dimension in [0, 1].
                        Returns ``None`` for an agent whose L3 was skipped
                        in every scenario (no judge available) so the
                        front-end can render an explicit "not measured"
                        state rather than misleading zeros.
  * ``failure_modes`` — flattened Top-N list of failures across L1
                        (status == "violated"), L2 ("partial"/"fail") and
                        L3 (normalised score < 0.5). Each item is
                        de-duplicated by (agent, scenario, finding_id) so
                        a single violation isn't double-counted.
  * ``low_confidence``— scenarios that need human attention: either
                        explicit ``needs_human_review``, a confidence
                        below the (configurable) threshold, or an L3-skip
                        meta flag.

The function is intentionally pure and depends only on the standard
library — keeping it usable from both the CLI report path and the
dashboard JSON endpoint without dragging in heavy deps.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _layered(detail: dict[str, Any]) -> dict[str, Any] | None:
    report = (detail.get("report") or {})
    return report.get("layered_report")


def _build_matrix(details: Iterable[dict[str, Any]]) -> dict[str, Any]:
    # agent -> instruction_id -> profile_id -> {score_sum, count, scenario_ids}
    raw: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"score_sum": 0.0, "count": 0, "scenario_ids": []}))
    )
    for d in details:
        layered = _layered(d)
        if layered is None:
            continue
        agent = d.get("agent_name", "unknown")
        scenario = d.get("scenario") or {}
        instr = str(scenario.get("instruction_id", "?"))
        profile = scenario.get("profile_id", "?")
        cell = raw[agent][instr][profile]
        cell["score_sum"] += float(layered.get("overall_score", 0.0))
        cell["count"] += 1
        cell["scenario_ids"].append(scenario.get("scenario_id", ""))

    matrix: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for agent, by_instr in raw.items():
        matrix[agent] = {}
        for instr, by_profile in by_instr.items():
            matrix[agent][instr] = {}
            for profile, cell in by_profile.items():
                count = cell["count"]
                # `count` can never be zero here because we only insert when
                # we have a layered_report, but guard anyway to make this
                # safe to call from tests that mock partial data.
                mean = cell["score_sum"] / count if count else 0.0
                matrix[agent][instr][profile] = {
                    "score_mean": round(mean, 2),
                    "count": count,
                    "scenario_ids": cell["scenario_ids"],
                }
    return matrix


def _build_radar(details: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Mean L3 dimension scores per agent, ``None`` per dimension when no L3 data."""
    # agent -> dimension -> list of scores in [0,1]
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Track agents we've seen at all so we can emit explicit None for
    # judge-less runs instead of dropping the agent from the radar entirely.
    seen_agents: set[str] = set()
    for d in details:
        layered = _layered(d)
        if layered is None:
            continue
        agent = d.get("agent_name", "unknown")
        seen_agents.add(agent)
        for f in layered.get("findings", []):
            if f.get("layer") != "L3":
                continue
            # finding_id is "L3.<dimension>"; split defensively.
            fid = f.get("finding_id", "")
            dim = fid.split(".", 1)[1] if "." in fid else fid
            bucket[agent][dim].append(float(f.get("score", 0.0)))

    radar: dict[str, dict[str, float | None]] = {}
    # Discover the full set of dimensions actually emitted, so callers
    # get a consistent shape even if one agent missed a dimension.
    all_dims = sorted({dim for by_dim in bucket.values() for dim in by_dim.keys()})
    for agent in seen_agents:
        if agent not in bucket or not bucket[agent]:
            # No L3 findings at all (judge unavailable for every scenario).
            radar[agent] = {dim: None for dim in all_dims}
            continue
        per_dim: dict[str, float | None] = {}
        for dim in all_dims:
            scores = bucket[agent].get(dim, [])
            per_dim[dim] = round(sum(scores) / len(scores), 4) if scores else None
        radar[agent] = per_dim
    return radar


def _is_failure(finding: dict[str, Any]) -> bool:
    """A finding counts as a failure for the failure-mode chart when:
    * L1 violations  → status == 'violated'
    * L2 step issues → status in {'partial', 'fail'}
    * L3 dimensions  → normalised score < 0.5  (≈ raw 1-5 score < 3)
    """
    layer = finding.get("layer")
    status = finding.get("status")
    if layer == "L1":
        return status == "violated"
    if layer == "L2":
        return status in ("partial", "fail")
    if layer == "L3":
        return float(finding.get("score", 0.0)) < 0.5
    return False


def _build_failure_modes(
    details: Iterable[dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, Any]] = {}
    for d in details:
        layered = _layered(d)
        if layered is None:
            continue
        agent = d.get("agent_name", "unknown")
        scenario = d.get("scenario") or {}
        scenario_id = scenario.get("scenario_id", "")
        seen_in_scenario: set[tuple[str, str]] = set()
        for f in layered.get("findings", []):
            if not _is_failure(f):
                continue
            key = (f.get("layer", "?"), f.get("finding_id", "?"))
            # De-dup per scenario so the same finding fired twice on a
            # single conversation doesn't pump the failure count.
            if (scenario_id, key) in seen_in_scenario:
                continue
            seen_in_scenario.add((scenario_id, key))
            entry = counts.setdefault(key, {
                "layer": key[0],
                "finding_id": key[1],
                "count": 0,
                "examples": [],
            })
            entry["count"] += 1
            if len(entry["examples"]) < 5:
                entry["examples"].append({
                    "agent_name": agent,
                    "scenario_id": scenario_id,
                    "rationale": f.get("rationale", ""),
                })
    ordered = sorted(counts.values(), key=lambda x: (-x["count"], x["finding_id"]))
    return ordered[:top_n]


def _build_low_confidence(
    details: Iterable[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in details:
        layered = _layered(d)
        if layered is None:
            continue
        confidence = float(layered.get("confidence", 1.0))
        needs_review = bool(layered.get("needs_human_review", False))
        l3_skipped = bool((layered.get("meta") or {}).get("l3_skipped", False))
        if not (needs_review or l3_skipped or confidence < threshold):
            continue
        scenario = d.get("scenario") or {}
        rows.append({
            "agent_name": d.get("agent_name", "unknown"),
            "scenario_id": scenario.get("scenario_id", ""),
            "instruction_id": scenario.get("instruction_id", ""),
            "profile_id": scenario.get("profile_id", ""),
            "overall_score": layered.get("overall_score"),
            "confidence": confidence,
            "needs_human_review": needs_review,
            "l3_skipped": l3_skipped,
            "inconsistency_flags": list(layered.get("inconsistency_flags") or []),
        })
    # Lowest-confidence first so the UI can render the most uncertain
    # rows at the top.
    rows.sort(key=lambda r: (r["confidence"], r["agent_name"], r["scenario_id"]))
    return rows


def aggregate(
    details: list[dict[str, Any]],
    *,
    low_confidence_threshold: float = 0.6,
    failure_top_n: int = 20,
) -> dict[str, Any]:
    """Aggregate a list of per-conversation detail dicts into report sections.

    ``details`` is the **list of detail dicts** produced by ``runner/cli.py``
    (each one having ``agent_name`` / ``scenario`` / ``report``). Passing
    the inner ``layered_report`` alone is not enough — we need the agent
    and scenario context to bucket things.
    """
    details = list(details)
    return {
        "matrix": _build_matrix(details),
        "radar": _build_radar(details),
        "failure_modes": _build_failure_modes(details, top_n=failure_top_n),
        "low_confidence": _build_low_confidence(details, threshold=low_confidence_threshold),
        "totals": {
            "scenarios": len(details),
            "agents": sorted({d.get("agent_name", "unknown") for d in details if _layered(d) is not None}),
            "low_confidence_threshold": low_confidence_threshold,
        },
    }


__all__ = ["aggregate"]
