const fallbackSummary = {
  aggregate: {
    scenario_count: 14,
    average_score: 49.56,
    min_score: 0,
    max_score: 100,
    failed_scenarios: [
      "1_cooperative_user",
      "1_busy_user",
      "1_skeptical_user",
      "1_refusing_user",
      "2_cooperative_user",
      "2_silent_user",
      "2_busy_user",
      "2_skeptical_user",
      "2_off_topic_user",
      "2_random_question_user",
      "2_driving_user",
    ],
  },
  rows: [
    { scenario_id: "1_cooperative_user", persona: "配合型用户", overall_score: 75.0, agent_turn_count: 4, violations: ["exceed_turn_length"] },
    { scenario_id: "1_silent_user", persona: "沉默型用户", overall_score: 100.0, agent_turn_count: 3, violations: [] },
    { scenario_id: "1_busy_user", persona: "忙碌型用户", overall_score: 70.0, agent_turn_count: 1, violations: ["exceed_turn_length"] },
    { scenario_id: "1_skeptical_user", persona: "质疑型用户", overall_score: 70.0, agent_turn_count: 4, violations: ["exceed_turn_length"] },
    { scenario_id: "1_off_topic_user", persona: "越界提问用户", overall_score: 90.0, agent_turn_count: 4, violations: [] },
    { scenario_id: "1_random_question_user", persona: "乱问问题用户", overall_score: 100.0, agent_turn_count: 4, violations: [] },
    { scenario_id: "1_refusing_user", persona: "拒绝执行用户", overall_score: 0.0, agent_turn_count: 2, violations: ["miss_primary_task", "miss_required_flow", "exceed_turn_length", "miss_end_call_condition"] },
    { scenario_id: "2_cooperative_user", persona: "配合型用户", overall_score: 50.0, agent_turn_count: 3, violations: ["exceed_turn_length"] },
    { scenario_id: "2_silent_user", persona: "沉默型用户", overall_score: 19.44, agent_turn_count: 4, violations: ["miss_primary_task", "exceed_turn_length"] },
    { scenario_id: "2_busy_user", persona: "忙碌型用户", overall_score: 16.67, agent_turn_count: 1, violations: ["miss_primary_task", "exceed_turn_length"] },
    { scenario_id: "2_skeptical_user", persona: "质疑型用户", overall_score: 50.0, agent_turn_count: 4, violations: ["exceed_turn_length"] },
    { scenario_id: "2_off_topic_user", persona: "越界提问用户", overall_score: 16.67, agent_turn_count: 4, violations: ["miss_primary_task", "exceed_turn_length"] },
    { scenario_id: "2_random_question_user", persona: "乱问问题用户", overall_score: 19.44, agent_turn_count: 3, violations: ["miss_primary_task", "exceed_turn_length"] },
    { scenario_id: "2_driving_user", persona: "开车中用户", overall_score: 16.67, agent_turn_count: 1, violations: ["miss_primary_task", "miss_required_flow"] },
  ],
};

function scoreClass(score) {
  if (score >= 80) return "score-high";
  if (score >= 50) return "score-mid";
  return "score-low";
}

function buildViolationStats(rows) {
  const stats = new Map();
  rows.forEach((row) => {
    row.violations.forEach((violation) => {
      stats.set(violation, (stats.get(violation) || 0) + 1);
    });
  });
  return [...stats.entries()].sort((a, b) => b[1] - a[1]);
}

function buildInsights(summary) {
  const rows = summary.rows;
  const task1 = rows.filter((row) => row.scenario_id.startsWith("1_"));
  const task2 = rows.filter((row) => row.scenario_id.startsWith("2_"));
  const avg = (items) => items.reduce((acc, item) => acc + item.overall_score, 0) / items.length;
  const violationStats = buildViolationStats(rows);

  return [
    `任务 1 平均分约 ${avg(task1).toFixed(2)}，明显高于任务 2 的 ${avg(task2).toFixed(2)}。`,
    `当前最常见的问题是 ${violationStats[0]?.[0] || "无"}，说明约束控制仍需加强。`,
    `“沉默型用户”和“乱问问题用户”在任务 1 上表现最好，说明当前 Agent 在简单推进类场景更稳。`,
    `“拒绝执行用户”和任务 2 的多个场景得分较低，说明复杂场景下主任务覆盖和结束策略还不够稳定。`,
  ];
}

function renderSummary(summary) {
  document.getElementById("metricScenarioCount").textContent = summary.aggregate.scenario_count;
  document.getElementById("metricAvgScore").textContent = summary.aggregate.average_score;
  document.getElementById("metricMaxScore").textContent = summary.aggregate.max_score;
  document.getElementById("metricMinScore").textContent = summary.aggregate.min_score;

  const sortedRows = [...summary.rows].sort((a, b) => b.overall_score - a.overall_score);
  const scoreBars = document.getElementById("scoreBars");
  scoreBars.innerHTML = sortedRows
    .map(
      (row) => `
      <div class="score-row">
        <div class="score-label">${row.scenario_id}</div>
        <div class="score-track">
          <div class="score-fill" style="width:${row.overall_score}%;"></div>
        </div>
        <div class="score-value">${row.overall_score}</div>
      </div>
    `
    )
    .join("");

  const issueList = document.getElementById("issueList");
  const issueStats = buildViolationStats(summary.rows);
  issueList.innerHTML = issueStats
    .map(
      ([issue, count]) => `
      <div class="issue-item">
        <strong>${issue}</strong>
        <span>触发 ${count} 次</span>
      </div>
    `
    )
    .join("");

  const tbody = document.getElementById("scenarioTableBody");
  tbody.innerHTML = summary.rows
    .map(
      (row) => `
      <tr>
        <td>${row.scenario_id}</td>
        <td>${row.persona}</td>
        <td><span class="score-chip ${scoreClass(row.overall_score)}">${row.overall_score}</span></td>
        <td>${row.agent_turn_count}</td>
        <td>
          <div class="violation-list">
            ${
              row.violations.length
                ? row.violations.map((item) => `<span class="violation-tag">${item}</span>`).join("")
                : `<span class="violation-tag">none</span>`
            }
          </div>
        </td>
      </tr>
    `
    )
    .join("");

  const insightList = document.getElementById("insightList");
  insightList.innerHTML = buildInsights(summary)
    .map((item) => `<li>${item}</li>`)
    .join("");
}

async function loadSummary() {
  try {
    const response = await fetch("./data/summary.json");
    if (!response.ok) throw new Error("fetch failed");
    return await response.json();
  } catch (error) {
    return fallbackSummary;
  }
}

loadSummary().then(renderSummary);
