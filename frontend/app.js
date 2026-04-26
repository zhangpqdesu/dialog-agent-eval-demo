'use strict';

const API = '';  // same origin

// ══════════════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════════════
document.querySelectorAll('.nav-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
    if (target === 'results') loadResults();
    if (target === 'personas') loadPersonaGrid();
    if (target === 'insights') loadInsights();
  });
});

// ══════════════════════════════════════════════
// SHARED STATE
// ══════════════════════════════════════════════
let _instructions = [];
let _personas = [];
let _results = null;
let _conversations = {};       // scenario_id -> conversation[]
let _tagStyles = [];           // current tags being edited in drawer
let _currentRunPersonas = {};  // personaId -> {name, turns[], scores[], done, score}
let _activeRunPersonaId = null;
let _scoreChartInst = null;
let _violationChartInst = null;

// ══════════════════════════════════════════════
// STATUS BAR
// ══════════════════════════════════════════════
function setStatus(state, text) {
  const dot = document.querySelector('.status-dot');
  const txt = document.querySelector('.status-text');
  dot.className = 'status-dot ' + state;
  txt.textContent = text;
}

// ══════════════════════════════════════════════
// TAB 1: LAB — LOAD CONFIG
// ══════════════════════════════════════════════
async function loadLabConfig() {
  const [instrRes, personaRes] = await Promise.all([
    fetch(API + '/api/instructions').then(r => r.json()).catch(() => ({ instructions: [] })),
    fetch(API + '/api/personas').then(r => r.json()).catch(() => ({ personas: [] })),
  ]);
  _instructions = instrRes.instructions || [];
  _personas = personaRes.personas || [];
  renderInstrList();
  renderPersonaCheckList();
}

function renderInstrList() {
  const el = document.getElementById('instrList');
  if (!_instructions.length) { el.innerHTML = '<div class="loading-placeholder">暂无指令数据</div>'; return; }
  el.innerHTML = _instructions.map(ins => `
    <div class="instr-card" data-id="${ins.instruction_id}" onclick="selectInstr('${ins.instruction_id}')">
      <div class="instr-card-id">指令 ${ins.instruction_id} · ${ins.domain || ''}</div>
      <div class="instr-card-role">${ins.role}</div>
      <div class="instr-card-task">${ins.task}</div>
    </div>
  `).join('');
  // auto-select first
  if (_instructions.length) selectInstr(_instructions[0].instruction_id);
}

function selectInstr(id) {
  document.querySelectorAll('.instr-card').forEach(c => c.classList.toggle('selected', c.dataset.id === id));
}

function getSelectedInstr() {
  const sel = document.querySelector('.instr-card.selected');
  return sel ? sel.dataset.id : (_instructions[0] && _instructions[0].instruction_id);
}

function renderPersonaCheckList() {
  const el = document.getElementById('personaCheckList');
  if (!_personas.length) { el.innerHTML = '<div class="loading-placeholder">暂无画像，请在「画像管理」中添加</div>'; return; }
  el.innerHTML = _personas.map(p => `
    <label class="persona-check-item">
      <input type="checkbox" value="${p.id}" />
      <span class="check-box">✓</span>
      <div>
        <div class="persona-check-label">${p.name}</div>
        <div class="persona-check-style">${(p.style || []).join('、') || '自然对话'}</div>
      </div>
    </label>
  `).join('');
}

function getSelectedPersonaIds() {
  return [...document.querySelectorAll('#personaCheckList input:checked')].map(i => i.value);
}

function getJudgeMode() {
  const checked = document.querySelector('input[name="judgeMode"]:checked');
  return checked ? checked.value : 'hybrid';
}

// ══════════════════════════════════════════════
// TAB 1: RUN EXPERIMENT
// ══════════════════════════════════════════════
async function startRun() {
  const instrId = getSelectedInstr();
  const personaIds = getSelectedPersonaIds();
  if (!instrId) { alert('请选择任务指令'); return; }
  if (!personaIds.length) { alert('请至少选择一个用户画像'); return; }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.querySelector('.run-btn-text').textContent = '实验中...';
  setStatus('running', '实验进行中');

  // init run UI
  _currentRunPersonas = {};
  _activeRunPersonaId = null;

  const runBody = document.getElementById('runArea');
  document.getElementById('labEmpty').style.display = 'none';
  runBody.style.display = 'flex';
  document.getElementById('personaRunTabs').innerHTML = '';
  document.getElementById('chatMessages').innerHTML = '';
  document.getElementById('scoreItems').innerHTML = '';
  document.getElementById('finalScoreReveal').style.display = 'none';
  document.getElementById('chatPersonaLabel').textContent = '等待开始...';
  document.getElementById('chatStatus').textContent = '运行中';

  // create persona tabs
  personaIds.forEach(pid => {
    const p = _personas.find(x => x.id === pid) || { id: pid, name: pid };
    _currentRunPersonas[pid] = { name: p.name, turns: [], scores: [], done: false };
    const tab = document.createElement('button');
    tab.className = 'persona-run-tab';
    tab.dataset.pid = pid;
    tab.textContent = p.name;
    tab.onclick = () => switchRunPersona(pid);
    document.getElementById('personaRunTabs').appendChild(tab);
  });

  // start run
  const res = await fetch(API + '/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ instruction_id: instrId, persona_ids: personaIds, judge_mode: getJudgeMode() }),
  }).then(r => r.json()).catch(e => { alert('启动失败: ' + e.message); return null; });

  if (!res || !res.run_id) {
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = '开始实验';
    setStatus('error', '启动失败');
    return;
  }

  // SSE stream
  const evtSource = new EventSource(API + '/api/run/' + res.run_id + '/stream');

  evtSource.addEventListener('persona_start', e => {
    const d = JSON.parse(e.data);
    switchRunPersona(d.persona_id);
  });

  evtSource.addEventListener('turn', e => {
    const d = JSON.parse(e.data);
    if (_activeRunPersonaId) {
      _currentRunPersonas[_activeRunPersonaId].turns.push(d);
    }
    appendChatTurn(d);
  });

  evtSource.addEventListener('score_item', e => {
    const d = JSON.parse(e.data);
    if (_activeRunPersonaId) {
      _currentRunPersonas[_activeRunPersonaId].scores.push(d);
    }
    appendScoreItem(d);
  });

  evtSource.addEventListener('persona_done', e => {
    const d = JSON.parse(e.data);
    const p = _currentRunPersonas[d.persona_id];
    if (p) {
      p.done = true;
      p.score = d.overall_score;
      p.ruleScore = d.rule_score;
      p.llmScore = d.llm_score;
    }
    const tab = document.querySelector(`.persona-run-tab[data-pid="${d.persona_id}"]`);
    if (tab) tab.classList.add('done');
    if (d.persona_id === _activeRunPersonaId) {
      showFinalScore(d);
    }
  });

  evtSource.addEventListener('all_done', () => {
    evtSource.close();
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = '开始实验';
    setStatus('done', '实验完成');
    document.getElementById('chatStatus').textContent = '完成';
  });

  evtSource.addEventListener('error', e => {
    try {
      const d = JSON.parse(e.data);
      console.error('SSE error:', d.message);
    } catch {}
    evtSource.close();
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = '开始实验';
    setStatus('error', '发生错误');
  });

  evtSource.onerror = () => {
    if (evtSource.readyState === EventSource.CLOSED) return;
    evtSource.close();
    btn.disabled = false;
    btn.querySelector('.run-btn-text').textContent = '开始实验';
    setStatus('done', '完成');
  };
}

function switchRunPersona(pid) {
  _activeRunPersonaId = pid;
  document.querySelectorAll('.persona-run-tab').forEach(t => t.classList.toggle('active', t.dataset.pid === pid));

  const p = _currentRunPersonas[pid];
  if (!p) return;

  document.getElementById('chatPersonaLabel').textContent = p.name + ' 对话';
  const msgs = document.getElementById('chatMessages');
  msgs.innerHTML = '';
  p.turns.forEach(t => appendChatTurn(t, false));

  const si = document.getElementById('scoreItems');
  si.innerHTML = '';
  p.scores.forEach(s => appendScoreItem(s, false));

  if (p.done) {
    showFinalScore({ overall_score: p.score, rule_score: p.ruleScore, llm_score: p.llmScore });
  } else {
    document.getElementById('finalScoreReveal').style.display = 'none';
  }
}

function appendChatTurn(d, animate = true) {
  const msgs = document.getElementById('chatMessages');
  const isAgent = d.role === 'agent';
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap' + (isAgent ? ' agent-wrap' : '');
  wrap.innerHTML = `
    <div class="bubble-avatar ${isAgent ? 'agent-avatar' : 'user-avatar'}">${isAgent ? '🤖' : '👤'}</div>
    <div class="bubble ${isAgent ? 'agent-bubble' : 'user-bubble'}"></div>
  `;
  msgs.appendChild(wrap);

  const bubble = wrap.querySelector('.bubble');
  if (animate && isAgent) {
    typewriter(bubble, d.text, 18);
  } else {
    bubble.textContent = d.text;
  }
  msgs.scrollTop = msgs.scrollHeight;
}

function typewriter(el, text, speed) {
  el.classList.add('typing-cursor');
  let i = 0;
  const iv = setInterval(() => {
    el.textContent = text.slice(0, ++i);
    if (i >= text.length) {
      clearInterval(iv);
      el.classList.remove('typing-cursor');
    }
  }, speed);
}

function appendScoreItem(d, animate = true) {
  const si = document.getElementById('scoreItems');
  const item = document.createElement('div');

  if (d.type === 'success') {
    const passed = d.passed;
    item.className = 'score-item ' + (passed ? 'pass' : 'fail');
    item.innerHTML = `
      <span class="score-item-icon">${passed ? '✅' : '❌'}</span>
      <span class="score-item-id">${d.criterion_id}</span>
      <span class="score-item-val ${passed ? 'pass' : 'fail'}">${passed ? '+' + (d.score || 0) : '0'}</span>
    `;
  } else {
    const triggered = d.triggered;
    item.className = 'score-item ' + (triggered ? 'fail' : 'pass');
    item.innerHTML = `
      <span class="score-item-icon">${triggered ? '⚠️' : '✅'}</span>
      <span class="score-item-id">${d.criterion_id} <span style="font-size:10px;color:var(--muted);">[${d.severity || ''}]</span></span>
      <span class="score-item-val ${triggered ? 'fail' : 'pass'}">${triggered ? '违规' : '通过'}</span>
    `;
  }

  si.appendChild(item);
  if (animate) {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => item.classList.add('visible'));
    });
    si.scrollTop = si.scrollHeight;
  } else {
    item.classList.add('visible');
  }
}

function showFinalScore(d) {
  const el = document.getElementById('finalScoreReveal');
  el.style.display = 'block';
  const num = document.getElementById('finalScoreNum');
  const breakdown = document.getElementById('finalScoreBreakdown');
  num.textContent = '--';
  let v = 0;
  const target = d.overall_score || 0;
  const iv = setInterval(() => {
    v = Math.min(v + target / 30, target);
    num.textContent = v.toFixed(1);
    if (v >= target) { clearInterval(iv); num.textContent = target.toFixed(1); }
  }, 30);

  const parts = [];
  if (d.rule_score != null) parts.push(`规则分 ${d.rule_score}`);
  if (d.llm_score != null)  parts.push(`LLM分 ${d.llm_score}`);
  breakdown.textContent = parts.join(' · ');
}

// ══════════════════════════════════════════════
// TAB 2: PERSONA MANAGEMENT
// ══════════════════════════════════════════════
async function loadPersonaGrid() {
  const res = await fetch(API + '/api/personas').then(r => r.json()).catch(() => ({ personas: [] }));
  _personas = res.personas || [];
  renderPersonaGrid();
  renderPersonaCheckList();
}

function renderPersonaGrid() {
  const grid = document.getElementById('personaGrid');
  if (!_personas.length) {
    grid.innerHTML = '<div class="loading-placeholder">还没有画像，点击「新建画像」开始</div>';
    return;
  }
  grid.innerHTML = _personas.map(p => `
    <div class="persona-card">
      <div class="persona-card-head">
        <div>
          <div class="persona-card-name">${p.name}</div>
          <div class="persona-card-id"># ${p.id} · 指令 ${p.instruction_id}</div>
        </div>
        <div class="persona-card-actions">
          <button class="icon-btn" title="编辑" onclick="openPersonaDrawer('${p.id}')">✎</button>
          <button class="icon-btn danger" title="删除" onclick="deletePersona('${p.id}', '${p.name}')">✕</button>
        </div>
      </div>
      <div class="persona-card-goal">${p.goal || '（未设置目标）'}</div>
      <div class="persona-style-tags">${(p.style || []).map(s => `<span class="style-tag">${s}</span>`).join('')}</div>
      <div class="persona-card-meta">
        <span>最大轮数: ${p.max_turns}</span>
        <span>开场: ${(p.initial_utterance || '').slice(0, 20)}...</span>
      </div>
    </div>
  `).join('');
}

// Drawer
function openPersonaDrawer(personaId) {
  document.getElementById('drawerOverlay').classList.add('open');
  document.getElementById('personaDrawer').classList.add('open');

  // populate instruction selector
  const sel = document.getElementById('pInstrId');
  sel.innerHTML = _instructions.map(i => `<option value="${i.instruction_id}">指令${i.instruction_id} · ${i.role}</option>`).join('');

  _tagStyles = [];
  renderTagChips();

  if (personaId) {
    const p = _personas.find(x => x.id === personaId);
    if (!p) return;
    document.getElementById('drawerTitle').textContent = '编辑画像';
    document.getElementById('editPersonaId').value = personaId;
    document.getElementById('pName').value = p.name || '';
    sel.value = p.instruction_id || '1';
    document.getElementById('pGoal').value = p.goal || '';
    _tagStyles = [...(p.style || [])];
    renderTagChips();
    document.getElementById('pUtterance').value = p.initial_utterance || '';
    document.getElementById('pMaxTurns').value = p.max_turns || 6;
  } else {
    document.getElementById('drawerTitle').textContent = '新建画像';
    document.getElementById('editPersonaId').value = '';
    document.getElementById('pName').value = '';
    document.getElementById('pGoal').value = '';
    document.getElementById('pUtterance').value = '你好，请说。';
    document.getElementById('pMaxTurns').value = 6;
  }
}

function closePersonaDrawer() {
  document.getElementById('drawerOverlay').classList.remove('open');
  document.getElementById('personaDrawer').classList.remove('open');
}

// tag input logic
document.getElementById('tagInputBox').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    const val = e.target.value.trim().replace(/,$/, '');
    if (val) { _tagStyles.push(val); renderTagChips(); }
    e.target.value = '';
  }
  if (e.key === 'Backspace' && !e.target.value && _tagStyles.length) {
    _tagStyles.pop();
    renderTagChips();
  }
});

function renderTagChips() {
  const container = document.getElementById('tagInputTags');
  container.innerHTML = _tagStyles.map((t, i) => `
    <span class="tag-chip">${t}<span class="tag-chip-del" onclick="removeTag(${i})">×</span></span>
  `).join('');
}

function removeTag(i) { _tagStyles.splice(i, 1); renderTagChips(); }

async function savePersona() {
  const id = document.getElementById('editPersonaId').value;
  const name = document.getElementById('pName').value.trim();
  if (!name) { alert('请填写画像名称'); return; }

  const payload = {
    name,
    instruction_id: document.getElementById('pInstrId').value || '1',
    goal: document.getElementById('pGoal').value.trim(),
    style: [..._tagStyles],
    initial_utterance: document.getElementById('pUtterance').value.trim() || '你好，请说。',
    max_turns: parseInt(document.getElementById('pMaxTurns').value) || 6,
    profile_id: 'custom',
  };

  if (id) {
    await fetch(API + '/api/personas/' + id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } else {
    await fetch(API + '/api/personas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  closePersonaDrawer();
  await loadPersonaGrid();
}

async function deletePersona(id, name) {
  if (!confirm(`确认删除画像「${name}」？`)) return;
  await fetch(API + '/api/personas/' + id, { method: 'DELETE' });
  await loadPersonaGrid();
}

// ══════════════════════════════════════════════
// TAB 3: RESULTS
// ══════════════════════════════════════════════
async function loadResults() {
  const data = await fetch(API + '/api/results').then(r => r.json()).catch(() => null);
  if (!data) return;
  _results = data;

  const agg = data.aggregate || {};
  animateCount('heroScore', agg.average_score || 0, 1);
  animateCount('metricCount', agg.scenario_count || 0, 0);
  animateCount('metricMax', agg.max_score || 0, 1);
  animateCount('metricMin', agg.min_score || 0, 1);
  animateCount('metricFailed', (agg.failed_scenarios || []).length, 0);
  animateRing(agg.average_score || 0);

  renderScoreChart(data.rows || []);
  renderViolationChart(data.rows || []);
  renderTable(data.rows || []);
  setupFilters();

  // store conversations for modal
  (data.rows || []).forEach(r => {
    if (r.conversation) _conversations[r.scenario_id] = r.conversation;
  });
}

function animateCount(id, target, decimals) {
  const el = document.getElementById(id);
  if (!el) return;
  let v = 0;
  const step = target / 40;
  const iv = setInterval(() => {
    v = Math.min(v + step, target);
    el.textContent = v.toFixed(decimals);
    if (v >= target) { clearInterval(iv); el.textContent = target.toFixed(decimals); }
  }, 25);
}

function animateRing(score) {
  const circ = 2 * Math.PI * 58;
  const fill = document.getElementById('ringFill');
  if (!fill) return;
  const offset = circ * (1 - Math.min(score, 100) / 100);
  requestAnimationFrame(() => {
    fill.style.transition = 'none';
    fill.setAttribute('stroke-dashoffset', circ);
    requestAnimationFrame(() => {
      fill.style.transition = 'stroke-dashoffset 1.2s ease';
      fill.setAttribute('stroke-dashoffset', offset);
    });
  });
}

function scoreClass(s) {
  if (s >= 80) return 'score-high';
  if (s >= 50) return 'score-mid';
  return 'score-low';
}

function renderScoreChart(rows) {
  const ctx = document.getElementById('scoreChart');
  if (!ctx) return;
  if (_scoreChartInst) { _scoreChartInst.destroy(); _scoreChartInst = null; }

  const labels = rows.map(r => r.persona || r.scenario_id);
  const scores = rows.map(r => r.overall_score || 0);
  const colors = scores.map(s => s >= 80 ? 'rgba(52,211,153,0.7)' : s >= 50 ? 'rgba(251,191,36,0.7)' : 'rgba(248,113,113,0.7)');

  _scoreChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: scores, backgroundColor: colors, borderRadius: 6, borderSkipped: false }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#7a8099', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { min: 0, max: 100, ticks: { color: '#7a8099', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
    },
  });
}

function renderViolationChart(rows) {
  const ctx = document.getElementById('violationChart');
  if (!ctx) return;
  if (_violationChartInst) { _violationChartInst.destroy(); _violationChartInst = null; }

  const m = new Map();
  rows.forEach(r => (r.violations || []).forEach(v => m.set(v, (m.get(v) || 0) + 1)));
  const sorted = [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!sorted.length) return;

  const palette = ['rgba(248,113,113,0.7)','rgba(251,191,36,0.7)','rgba(79,127,255,0.7)','rgba(124,92,252,0.7)','rgba(52,211,153,0.7)','rgba(251,146,60,0.7)'];

  _violationChartInst = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: sorted.map(e => e[0]),
      datasets: [{ data: sorted.map(e => e[1]), backgroundColor: palette, borderWidth: 2, borderColor: '#131620' }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#7a8099', font: { size: 11 }, boxWidth: 12 } },
      },
    },
  });
}

function renderTable(rows) {
  const tbody = document.getElementById('scenarioTbody');
  if (!tbody) return;
  tbody.innerHTML = rows.map(r => `
    <tr data-instr="${r.instruction_id}">
      <td><code style="font-size:12px;color:var(--muted)">${r.scenario_id}</code></td>
      <td>${r.persona || '--'}</td>
      <td><span class="score-chip ${scoreClass(r.overall_score)}">${(r.overall_score || 0).toFixed(1)}</span></td>
      <td>${r.rule_score != null ? r.rule_score.toFixed(1) : '--'}</td>
      <td>${r.llm_score != null ? r.llm_score.toFixed(1) : '--'}</td>
      <td>${r.agent_turn_count || 0}</td>
      <td>
        <div class="vtag-wrap">
          ${r.violations && r.violations.length
            ? r.violations.map(v => `<span class="vtag">${v}</span>`).join('')
            : '<span class="vtag vtag-none">无违规</span>'}
        </div>
      </td>
      <td>
        <button class="replay-btn" onclick="openConvModal('${r.scenario_id}')">回放</button>
      </td>
    </tr>
  `).join('');
}

function setupFilters() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const f = btn.dataset.filter;
      document.querySelectorAll('#scenarioTbody tr').forEach(tr => {
        tr.classList.toggle('hidden', f !== 'all' && tr.dataset.instr !== f);
      });
    });
  });
}

// ══════════════════════════════════════════════
// CONVERSATION MODAL
// ══════════════════════════════════════════════
document.getElementById('modalClose').onclick = () => document.getElementById('modalOverlay').classList.remove('open');

function openConvModal(scenarioId) {
  const conv = _conversations[scenarioId];
  const modal = document.getElementById('modalOverlay');
  const body = document.getElementById('modalBody');
  document.getElementById('modalTitle').textContent = '对话回放 · ' + scenarioId;

  if (!conv || !conv.length) {
    body.innerHTML = '<p style="color:var(--muted);font-size:14px;">暂无对话记录（请先运行实验）</p>';
  } else {
    body.innerHTML = conv.map(turn => {
      const isAgent = turn.role === 'agent';
      return `
        <div class="bubble-wrap ${isAgent ? 'agent-wrap' : ''}">
          <div class="bubble-avatar ${isAgent ? 'agent-avatar' : 'user-avatar'}">${isAgent ? '🤖' : '👤'}</div>
          <div class="bubble ${isAgent ? 'agent-bubble' : 'user-bubble'}">${turn.text || ''}</div>
        </div>
      `;
    }).join('');
  }

  modal.classList.add('open');
}

// ══════════════════════════════════════════════
// TAB 4: INSIGHTS
// ══════════════════════════════════════════════
function loadInsights() {
  const grid = document.getElementById('insightsGrid');
  if (!grid) return;

  const findings = _results ? buildInsights(_results) : defaultInsights();
  grid.innerHTML = findings.map(f => `
    <div class="insight-card">
      <div class="insight-icon">${f.icon}</div>
      <div class="insight-text">${f.html}</div>
    </div>
  `).join('');
}

function buildInsights(data) {
  const rows = data.rows || [];
  const agg = data.aggregate || {};
  const violations = new Map();
  rows.forEach(r => (r.violations || []).forEach(v => violations.set(v, (violations.get(v) || 0) + 1)));
  const topV = [...violations.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
  const avgScore = agg.average_score || 0;
  const failCount = (agg.failed_scenarios || []).length;

  return [
    {
      icon: '📊',
      html: `<strong>整体得分 ${avgScore.toFixed(1)}</strong>，共 ${agg.scenario_count} 个场景，其中 <strong>${failCount}</strong> 个存在违规行为。`,
    },
    {
      icon: '🔴',
      html: topV.length
        ? `最常见违规类型：<strong>${topV[0][0]}</strong>（${topV[0][1]}次）${topV[1] ? '、<strong>' + topV[1][0] + '</strong>（' + topV[1][1] + '次）' : ''}，需重点优化。`
        : '暂无违规统计数据，请先运行实验。',
    },
    {
      icon: '🎯',
      html: '规则评分与 LLM 语义评分<strong>双层互补</strong>，规则保证精确性，LLM 捕捉对话质量和情感流畅度。',
    },
    {
      icon: '💡',
      html: '建议在<strong>拒绝型</strong>和<strong>质疑型</strong>用户画像上重点调优，这类场景最容易触发超轮数和任务遗漏。',
    },
  ];
}

function defaultInsights() {
  return [
    { icon: '🤖', html: '<strong>三角色自动评测</strong>：Dialog Agent × User Simulator × Evaluator，全程无需人工干预。' },
    { icon: '⚡', html: '<strong>DeepSeek API 驱动</strong>：低成本大规模测试，一次实验涵盖全部画像类型。' },
    { icon: '📏', html: '<strong>双层评分</strong>：规则层（关键词/流程）+ LLM 层（语义质量），综合分更可信。' },
    { icon: '🔧', html: '<strong>画像动态配置</strong>：无需改代码，在「画像管理」中实时增删改，立即生效。' },
  ];
}

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
(async function init() {
  await loadLabConfig();
  // pre-load results in background so results tab is fast
  fetch(API + '/api/results').then(r => r.json()).then(data => {
    _results = data;
    if (data && data.rows) {
      data.rows.forEach(r => { if (r.conversation) _conversations[r.scenario_id] = r.conversation; });
    }
  }).catch(() => {});
})();
