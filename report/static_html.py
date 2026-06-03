"""Single-file static HTML report for an evaluation run.

The report is deliberately a single self-contained ``.html`` file with
ECharts loaded from a CDN — small enough to ship as an artefact in
CI/Slack and viewable without spinning up the dashboard server. The
report layout matches §4.6 of the design doc:

  * KPI 卡  — overall mean score, scenario count, low-confidence count,
              L1 violation count.
  * 矩阵    — agent × instruction × profile heatmap of mean scores.
  * 雷达    — per-agent mean of each L3 dimension. Agents whose L3 was
              skipped are drawn as flat zero with a "not measured"
              label so the absence is visible rather than misleading.
  * 失败模式 — Top-N bar chart over L1+L2+L3 failures.
  * 低置信度 — table of scenarios flagged for human review.
  * 详情     — only the worst ``max_detail`` scenarios are inlined to
              keep the HTML small even when the run has hundreds of
              conversations.

The function is pure: it accepts already-aggregated data and per-
conversation details and returns a string. No I/O. Tests exercise it
directly; the CLI just writes the returned string to ``report.html``.
"""
from __future__ import annotations

import html
import json
from typing import Any


_ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _kpi_cards(summary: dict[str, Any], aggregate_dict: dict[str, Any]) -> str:
    results = summary.get("results") or []
    scenario_count = len(results)
    mean_score = (
        round(sum(r.get("overall_score", 0.0) for r in results) / scenario_count, 2)
        if scenario_count else 0.0
    )
    low_conf = len(aggregate_dict.get("low_confidence") or [])
    l1_total = sum(
        row["count"] for row in (aggregate_dict.get("failure_modes") or [])
        if row.get("layer") == "L1"
    )
    cards = [
        ("平均总分", mean_score),
        ("场景数", scenario_count),
        ("低置信度", low_conf),
        ("L1 违规", l1_total),
    ]
    items = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{_escape(label)}</div>'
        f'<div class="kpi-value">{_escape(value)}</div></div>'
        for label, value in cards
    )
    return f'<section class="kpi-row">{items}</section>'


def _pick_top_details(
    details: list[dict[str, Any]],
    *,
    max_detail: int,
) -> list[dict[str, Any]]:
    """Pick the worst-N scenarios by (review-flagged, then score asc)."""
    def sort_key(d: dict[str, Any]) -> tuple[int, float]:
        layered = (d.get("report") or {}).get("layered_report") or {}
        return (
            0 if layered.get("needs_human_review") else 1,
            float(layered.get("overall_score", 0.0)),
        )
    return sorted(details, key=sort_key)[:max_detail]


def _detail_panels(details: list[dict[str, Any]]) -> str:
    panels = []
    for d in details:
        layered = (d.get("report") or {}).get("layered_report") or {}
        scenario = d.get("scenario") or {}
        conv_html = "".join(
            f'<div class="turn turn-{_escape(t.get("role", "?"))}">'
            f'<span class="role">{_escape(t.get("role", "?"))}</span>'
            f'<span class="text">{_escape(t.get("text", ""))}</span></div>'
            for t in (d.get("conversation") or [])
        )
        flag_html = ""
        flags = layered.get("inconsistency_flags") or []
        if flags:
            flag_html = '<ul class="flags">' + "".join(
                f"<li>{_escape(f)}</li>" for f in flags
            ) + "</ul>"
        findings_rows = []
        for f in layered.get("findings") or []:
            findings_rows.append(
                "<tr>"
                f"<td>{_escape(f.get('layer'))}</td>"
                f"<td>{_escape(f.get('finding_id'))}</td>"
                f"<td>{_escape(f.get('status'))}</td>"
                f"<td>{_escape(round(float(f.get('score', 0.0)), 3))}</td>"
                f"<td>{_escape(f.get('rationale', ''))}</td>"
                "</tr>"
            )
        findings_html = (
            "<table class='findings'><thead><tr>"
            "<th>层</th><th>finding_id</th><th>状态</th><th>分数</th><th>解释</th>"
            "</tr></thead><tbody>"
            + "".join(findings_rows) + "</tbody></table>"
        )
        panels.append(
            f'<details class="case">'
            f'<summary><strong>{_escape(d.get("agent_name"))}</strong> · '
            f'{_escape(scenario.get("scenario_id"))} · '
            f'overall={_escape(layered.get("overall_score"))} · '
            f'confidence={_escape(layered.get("confidence"))}</summary>'
            f'<div class="case-body">'
            f'{flag_html}'
            f'<div class="conversation">{conv_html}</div>'
            f'{findings_html}'
            f'</div></details>'
        )
    return "<section class='cases'><h2>场景详情（Top）</h2>" + "".join(panels) + "</section>"


def _low_confidence_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    body = "".join(
        "<tr>"
        f"<td>{_escape(r.get('agent_name'))}</td>"
        f"<td>{_escape(r.get('scenario_id'))}</td>"
        f"<td>{_escape(r.get('instruction_id'))}</td>"
        f"<td>{_escape(r.get('profile_id'))}</td>"
        f"<td>{_escape(r.get('overall_score'))}</td>"
        f"<td>{_escape(round(float(r.get('confidence', 0.0)), 3))}</td>"
        f"<td>{_escape('是' if r.get('needs_human_review') else '否')}</td>"
        f"<td>{_escape('是' if r.get('l3_skipped') else '否')}</td>"
        "</tr>"
        for r in rows
    )
    return (
        "<section><h2>低置信度场景</h2>"
        "<table class='low-conf'><thead><tr>"
        "<th>agent</th><th>scenario</th><th>instruction</th><th>profile</th>"
        "<th>分数</th><th>置信度</th><th>需复核</th><th>L3 跳过</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></section>"
    )


def render(
    summary: dict[str, Any],
    per_conversation_details: list[dict[str, Any]],
    aggregate_dict: dict[str, Any],
    *,
    max_detail: int = 50,
) -> str:
    """Render the run as a single self-contained HTML string."""
    top_details = _pick_top_details(per_conversation_details, max_detail=max_detail)

    # Embed the data needed by the JS chart code as JSON. Any future
    # front-end iteration can read straight from these blobs without
    # re-running aggregation.
    data_blob = {
        "matrix": aggregate_dict.get("matrix", {}),
        "radar": aggregate_dict.get("radar", {}),
        "failure_modes": aggregate_dict.get("failure_modes", []),
        "totals": aggregate_dict.get("totals", {}),
        "run_name": summary.get("run_name", ""),
    }

    # ECharts initialisation is intentionally tolerant of empty data — the
    # report should still render usefully if a run produced zero scoring
    # rows (e.g. all adapters errored out).
    js = """
    const DATA = __DATA__;
    function renderMatrix() {
      const el = document.getElementById('chart-matrix');
      if (!el) return;
      const matrix = DATA.matrix || {};
      const agents = Object.keys(matrix).sort();
      const cellSet = new Set();
      const rows = [];
      agents.forEach(agent => {
        Object.entries(matrix[agent] || {}).forEach(([instr, byProfile]) => {
          Object.entries(byProfile || {}).forEach(([profile, cell]) => {
            const colKey = instr + ' / ' + profile;
            cellSet.add(colKey);
            rows.push([colKey, agent, cell.score_mean, cell.count, (cell.scenario_ids||[]).join(',')]);
          });
        });
      });
      const cols = Array.from(cellSet).sort();
      const data = rows.map(r => [cols.indexOf(r[0]), agents.indexOf(r[1]), r[2], r[3], r[4]]);
      const chart = echarts.init(el);
      chart.setOption({
        tooltip: { formatter: p => `${p.value[4]}<br/>mean=${p.value[2]} (n=${p.value[3]})` },
        grid: { left: 100, right: 40, top: 40, bottom: 100 },
        xAxis: { type: 'category', data: cols, axisLabel: { rotate: 30 } },
        yAxis: { type: 'category', data: agents },
        visualMap: { min: 0, max: 100, calculable: true, orient: 'horizontal', left: 'center', bottom: 10 },
        series: [{ type: 'heatmap', data, label: { show: true, formatter: p => p.value[2] } }],
      });
    }
    function renderRadar() {
      const el = document.getElementById('chart-radar');
      if (!el) return;
      const radar = DATA.radar || {};
      const agents = Object.keys(radar).sort();
      const dims = Array.from(new Set(agents.flatMap(a => Object.keys(radar[a] || {})))).sort();
      if (dims.length === 0) { el.innerHTML = '<p class="empty">无 L3 数据</p>'; return; }
      const indicators = dims.map(d => ({ name: d, max: 1 }));
      const series = agents.map(agent => {
        const vals = dims.map(d => {
          const v = (radar[agent] || {})[d];
          return v == null ? 0 : v;
        });
        const missing = dims.some(d => (radar[agent] || {})[d] == null);
        return { name: agent + (missing ? ' (部分未评)' : ''), value: vals };
      });
      const chart = echarts.init(el);
      chart.setOption({
        tooltip: {},
        legend: { data: series.map(s => s.name), bottom: 0 },
        radar: { indicator: indicators },
        series: [{ type: 'radar', data: series }],
      });
    }
    function renderFailures() {
      const el = document.getElementById('chart-failures');
      if (!el) return;
      const rows = DATA.failure_modes || [];
      if (rows.length === 0) { el.innerHTML = '<p class="empty">未发现失败</p>'; return; }
      const labels = rows.map(r => r.layer + ' · ' + r.finding_id);
      const counts = rows.map(r => r.count);
      const chart = echarts.init(el);
      chart.setOption({
        tooltip: {},
        grid: { left: 220, right: 40, top: 20, bottom: 30 },
        xAxis: { type: 'value' },
        yAxis: { type: 'category', data: labels.reverse(), axisLabel: { fontSize: 11 } },
        series: [{ type: 'bar', data: counts.slice().reverse() }],
      });
    }
    window.addEventListener('load', () => { renderMatrix(); renderRadar(); renderFailures(); });
    """
    js = js.replace(
        "__DATA__",
        # JSON.stringify alone doesn't protect a script tag: a value like
        # "</script>" would close the tag and let attacker content escape.
        # Escape "</" -> "<\/" (still valid JSON, parser-equivalent) so
        # the browser tokenizer never sees a closing tag mid-payload.
        json.dumps(data_blob, ensure_ascii=False).replace("</", "<\\/"),
    )

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 24px; color: #1f2328; }
    h1 { margin-bottom: 0; }
    .meta { color: #57606a; margin-top: 4px; margin-bottom: 16px; }
    .kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
    .kpi-card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 18px; min-width: 120px; }
    .kpi-label { color: #57606a; font-size: 12px; }
    .kpi-value { font-size: 24px; font-weight: 600; margin-top: 4px; }
    section { margin-bottom: 32px; }
    .chart { width: 100%; height: 360px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    table th, table td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }
    table th { background: #f6f8fa; }
    .case { border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 8px; padding: 8px 12px; }
    .case summary { cursor: pointer; }
    .case-body { margin-top: 8px; }
    .conversation { background: #f6f8fa; padding: 8px; border-radius: 4px; margin-bottom: 8px; }
    .turn { display: flex; gap: 8px; margin-bottom: 4px; }
    .turn .role { color: #57606a; min-width: 48px; font-size: 12px; }
    .flags { color: #9a6700; }
    .empty { color: #57606a; }
    """

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Eval Report — {_escape(summary.get('run_name', 'run'))}</title>
<script src="{_ECHARTS_CDN}"></script>
<style>{css}</style>
</head>
<body>
<h1>{_escape(summary.get('run_name', 'run'))}</h1>
<p class="meta">started_at={_escape(summary.get('started_at', ''))} ·
finished_at={_escape(summary.get('finished_at', ''))} ·
seed={_escape(summary.get('seed', ''))} ·
judge={_escape(summary.get('judge_provider', ''))}
({_escape('available' if summary.get('judge_available') else 'unavailable')})</p>
{_kpi_cards(summary, aggregate_dict)}
<section><h2>矩阵：agent × instruction/profile</h2><div id="chart-matrix" class="chart"></div></section>
<section><h2>雷达：L3 各维度</h2><div id="chart-radar" class="chart"></div></section>
<section><h2>失败模式 Top-N</h2><div id="chart-failures" class="chart"></div></section>
{_low_confidence_table(aggregate_dict.get('low_confidence') or [])}
{_detail_panels(top_details)}
<script>{js}</script>
</body>
</html>
"""
    return html_doc


__all__ = ["render"]
