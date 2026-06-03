# 对话 Agent 指令遵循自动评测平台 — 设计文档

- 日期：2026-06-03
- 主轴：**工程完整**（B），融合方法新颖（A）/ 可解释性（C）/ 覆盖与可靠（D）作为支撑
- 交付周期：标准型 2–3 周
- 状态：设计稿，待用户审阅

---

## 1. 背景与目标

### 1.1 任务背景
履约数字人外呼场景中，系统自动发起与用户的通话，对话模型需根据预设指令完成具体任务。指令包含复杂流程、多类约束与禁用条款，人工评估成本高且难以量化。

### 1.2 设计目标
构建一套面向**指令遵循效果**的自动化评测平台，满足：
- **可解释**：每个分数都能追溯到对话中具体的一句话与判定理由
- **可量化**：总分、维度分、通过率、失败模式分布、置信度均可数值化
- **可靠**：多次采样 + 一致性检查 + 置信度告警，避免单次 LLM judge 偶然性
- **可扩展**：吃任意新指令、接任意新被测 Agent、通过一个配置文件驱动全流程

### 1.3 非目标
- 不解决「电话语音 ASR/TTS」相关问题，输入输出统一为文本
- 不替代人工最终评审，对低置信度场景仍标记 `needs_human_review`
- 不做模型训练 / 微调，仅评测

---

## 2. 与现有 demo 的关系

### 2.1 保留
- 目录骨架（`agent/`、`simulator/`、`evaluator/`、`runner/`、`frontend/`、`scripts/`、`data/`）
- schema 分层（基础版 + 评测版）
- 双模可切换的思路（rule / LLM）
- `api_server.py` 与 `frontend/` 作为 Dashboard 起点
- LLM 客户端封装（`llm/kimi_client.py`、`llm/deepseek_client.py`）

### 2.2 重写
- `scripts/extract_instruction_schema.py`：从 1035 行规则抽取改为「LLM 抽取 + JSON Schema 校验 + 失败回环」
- `simulator/user_simulator.py` 与 `simulator/llm_user_simulator.py`：合并为「LLM 生成话术 + 显式状态机/槽位约束」的混合模拟器，并新增红队 persona
- `evaluator/auto_scorer.py` 与 `evaluator/llm_judge.py`：重构为分层评分（L1/L2/L3）+ 多采样投票 + 证据引用

### 2.3 新增
- `config/`：YAML 配置驱动入口
- `agent/adapters/`：被测 Agent 适配器抽象（内置 LLM / HTTP / OpenAI 兼容 / 离线日志）
- `evaluator/aggregator.py`：跨场景聚合与失败模式归类
- `report/`：静态 HTML 报告生成器（ECharts CDN）
- Dashboard 路由扩展：实时进度 + 对比模式

---

## 3. 总体架构

六层结构，数据自上而下流动，报告自下而上汇聚。

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 配置层  eval_config.yaml                                  │
│    instructions / personas / agents_under_test / scoring     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. 指令结构化层  Instruction Structuring                     │
│    raw text/excel  →  LLM extract  →  schema validate  →     │
│    structured instruction (with success/failure criteria)    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. 被测 Agent 适配层  Agent Adapter                          │
│    BuiltinLLMAgent / HTTPAgent / OpenAICompatAgent /         │
│    OfflineLogAgent  ── 统一 respond(history) → str           │
└──────────────────────────────────────────────────────────────┘
                              ↓ ←─── 4. 用户模拟器
┌──────────────────────────────────────────────────────────────┐
│ 4. 用户模拟器层  User Simulator                              │
│    HybridSimulator = LLM(话术) + StateMachine(状态/槽位/退出)│
│    + RedTeamPersona（诱导超字数/超范围/漏流程的对抗用户）    │
└──────────────────────────────────────────────────────────────┘
                              ↓ 对话轨迹
┌──────────────────────────────────────────────────────────────┐
│ 5. 评分层  Scoring                                           │
│    L1 RuleGate (硬约束 0/1)                                  │
│    L2 FlowCoverage (规则取证 + LLM 复核)                     │
│    L3 SemanticJudge (多维度 × N 次采样投票 + 置信度)         │
│    EvidenceLinker (每条结论 → turn_index + 摘录 + 理由)      │
└──────────────────────────────────────────────────────────────┘
                              ↓ per-conversation report
┌──────────────────────────────────────────────────────────────┐
│ 6. 报告层  Reporting                                         │
│    Aggregator → matrix / radar / failure-mode top-N          │
│    StaticHTMLRenderer (ECharts)                              │
│    Dashboard (FastAPI + 实时进度 + Agent 对比)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 模块详细设计

### 4.1 配置层

**入口**：`eval_config.yaml`

```yaml
run_name: rider_ops_kimi_vs_deepseek_v1
instructions:
  source: data/processed/dialog_instruction_eval_examples.json
  filter_ids: [1, 2]            # 可选
personas:
  source: data/personas.json
  include_red_team: true
agents_under_test:
  - name: kimi-k2
    type: builtin_llm
    model: kimi-k2-turbo-preview
  - name: deepseek-chat
    type: builtin_llm
    model: deepseek-chat
  - name: my-prod-agent
    type: http
    endpoint: http://localhost:9000/respond
scoring:
  l3_samples: 3                  # 多采样次数
  judge_model: kimi-k2-turbo-preview
  confidence_threshold: 0.6
output:
  dir: outputs/{run_name}
  formats: [json, html]
```

**CLI**：`python -m runner.cli --config eval_config.yaml`

**约束**：所有 LLM 调用受 `seed` 与 `temperature` 控制；`run_name` 决定输出目录，结果天然版本化。

### 4.2 指令结构化层

**输入**：原始 Excel / Markdown 指令文本
**输出**：评测版 JSON（保留现有 `dialog_instruction_eval.schema.json` 字段）

**新流程**：
1. **LLM 抽取**：单次调用，prompt 内嵌目标 JSON Schema 与示例，让模型直出结构化结果
2. **Schema 校验**：`jsonschema.validate`；失败则把错误信息回灌给 LLM 让其修复（最多 2 轮）
3. **可选人工 review**：在 `data/processed/` 旁生成 `*.review.md`，列出抽取出的成功标准 / 失败条件，方便快速人眼过

**收益**：摆脱对原 Excel 标题结构的硬编码，新指令零代码可入。

### 4.3 被测 Agent 适配层

**抽象基类**：
```python
class AgentAdapter:
    def respond(self, example: dict, scenario: dict,
                history: list[dict]) -> str: ...
```

**四类实现**：
- `BuiltinLLMAgent`：复用现有 `APIAgent`（Kimi/DeepSeek）
- `HTTPAgent`：POST JSON `{instruction, history}` 到外部 endpoint，期望返回 `{reply: str}`
- `OpenAICompatAgent`：把指令拼成 system prompt，按 OpenAI Chat Completions 协议调用
- `OfflineLogAgent`：从已存日志中按顺序"回放"每一轮 agent 回复——评分不再依赖在线 Agent，可吃真实生产对话

### 4.4 用户模拟器层

**核心数据结构**：场景（保留现有 schema）+ 显式 `goal_slots`（需要确认的信息字段）+ `exit_conditions`（终止条件 DSL）

**HybridSimulator.reply(agent_message)** 流程：
1. **状态机更新**：根据 agent_message 关键词/槽位填充检查，决定当前状态、`goal_progress`
2. **LLM 生成话术**：把当前状态、角色 persona、目标、未填槽位、风格作为 prompt context，要求 LLM 生成自然 user_reply
3. **退出判定**：若满足 `exit_conditions`（如「agent 已确认所有必填槽位」、「agent 主动结束」、「达到 max_turns」），返回 `finished=True`
4. 同时返回结构化字段：`{user_reply, state_id, goal_progress, finished, exit_reason}`

**Persona 库扩展**（在现有 6 类基础上补齐）：
- 基础类：配合 / 沉默 / 忙碌 / 质疑 / 跑题 / 拒绝
- 边界类：情绪化、模糊回答、反复改主意、信息不全
- **红队类**（新增）：
  - `RT_lure_oversize`：诱导 agent 长篇大论以触发超字数
  - `RT_lure_off_scope`：连续追问知识范围外问题
  - `RT_lure_skip_step`：佯装已了解信息，引诱 agent 跳过必经流程
  - `RT_force_endcall_omission`：营造"该挂断但不显眼"的场景

红队 persona 与"测试用例"一一对应，自然形成失败模式覆盖矩阵。

### 4.5 评分层

#### L1 — 硬约束规则（RuleGate）
确定性、零歧义的 0/1 判定。例：
- `max_chars_per_turn`：扫描每个 agent turn 字符数
- `forbidden_expressions`：禁用词出现
- `must_endcall_in_driving`：driving_user 场景下最后一轮是否含挂断意图
- `must_use_opening_line`：首轮是否使用规定开场白
- `out_of_scope_fallback_used`：跑题场景是否使用规定兜底话术

每条结果：`{rule_id, passed: bool, evidence: [{turn_index, excerpt}], reason}`
**违反 L1 = 硬扣分项**（在总分上有固定权重 + 报告中独立标红），但不直接归零，避免一票否决掩盖其它信息。

#### L2 — 流程覆盖（FlowCoverage）
**两步法**：
1. **规则取证**：对每个 `call_flow.step`，扫描 agent 全文，找到候选命中轮（关键词 + 句子语义粗筛）
2. **LLM 复核**：把候选轮 + 步骤描述给 LLM，问"agent 是否真的在执行/完成了这个步骤"，返回 `pass | partial | fail` + 引用

每条结果：`{step_id, status, evidence_turn_indices, llm_reason}`

#### L3 — 语义质量（SemanticJudge）
LLM Judge 按固定**多维度评分卡**（每维 1–5 分）：
- `naturalness`：表达自然度
- `accuracy`：信息事实准确度（基于 knowledge_points）
- `empathy_handling`：对用户情绪 / 拒绝 / 催促的处理
- `goal_pursuit`：是否有效推进任务目标
- `constraint_awareness`：对约束（语气、长度、范围）的隐性感知

**采样投票**：每个维度独立调用 LLM Judge **N 次**（默认 3），取中位数作为该维度得分，标准差作为该维度**置信度**（`confidence = 1 - normalized_std`）。

每条结果：`{dimension, score, confidence, samples: [...], rationale, evidence_turns}`

#### 总分聚合
```
overall = 100 - L1_penalty
        + L2_weight * L2_pass_ratio
        + L3_weight * weighted_mean(L3_dimension_scores)
```
权重默认 `L1_penalty_per_violation=10, L2_weight=40, L3_weight=60` —— 配置可调。

#### EvidenceLinker
**所有**（L1/L2/L3）结论统一带：
```json
{
  "evidence": [
    {"turn_index": 3, "role": "agent",
     "excerpt": "...原文摘录...",
     "reason": "为什么这一句触发了此判定"}
  ]
}
```
报告渲染时高亮对应对话气泡，做到「点击分数 → 跳转对话证据」。

#### 矛盾检查（轻量版，stretch 可加）
若 L1 触发 `miss_required_flow` 但 L2 给出 `pass`，写入 `inconsistency_flags`，报告中提示 `needs_human_review`。

### 4.6 报告层

#### 聚合器（Aggregator）
输入：所有单场景报告
输出：
- 矩阵：`agent × instruction × persona → overall_score`
- 维度雷达：每个 agent 在 L3 五维度的平均分
- 失败模式 Top-N：跨场景统计哪些 L1 rule / L2 step / L3 低分维度最常出问题
- 置信度分布：低置信度场景列表

#### 静态 HTML 报告
- 单文件 `report.html`，内联数据 + 引用 ECharts CDN
- 模块：总览卡片 / 矩阵热力图（点击下钻）/ 雷达图（多 agent 叠加）/ 失败模式条形图 / 单场景详情（对话回放 + 证据高亮）/ 低置信度列表
- 可离线打开、可发邮件、可截图

#### Dashboard（FastAPI）
- 现有 `api_server.py` 扩展：
  - `POST /runs` 启动一次评测，返回 `run_id`
  - `GET /runs/{run_id}/progress` SSE 推送实时进度
  - `GET /runs/{run_id}/report` 返回聚合报告 JSON
  - `GET /runs/{run_id}/compare?other={run_id}` 对比模式
- 前端 `frontend/`：
  - 启动页（选配置 / 上传 yaml）
  - 实时页（进度条 + 当前对话流）
  - 报告页（同 HTML 报告）
  - 对比页（两个 run 并排展示矩阵 + 雷达差异）

---

## 5. 数据流示例（单场景）

```
config → load instruction "1" + persona "1_skeptical_user"
       → AgentAdapter("kimi-k2") + HybridSimulator(scenario)
       → loop:
            user_msg = sim.start() / sim.reply(agent_msg)
            agent_msg = adapter.respond(history)
            until finished
       → ConversationLog
       → Scorer.score(L1, L2, L3 × N samples)
       → per_conversation_report.json (含证据)
       → Aggregator 累加
       → 全部完成后 → report.html + Dashboard 推送
```

---

## 6. 关键质量特性落地

| 特性 | 实现位置 | 量化指标 |
|---|---|---|
| 可解释 | EvidenceLinker，所有结论带 turn_index + 摘录 + 理由 | 100% 结论可追溯 |
| 可量化 | 总分 / L1 违规数 / L2 通过率 / L3 五维度分 / 置信度 / 失败模式分布 | 全部数值 |
| 可靠 | L3 多采样投票 + 置信度 + L1/L2/L3 矛盾标记 | 每场景输出 `confidence ∈ [0, 1]` |
| 可复现 | 配置版本化 + `seed` + 输出目录含 `run_name` | 同配置二次运行差异可量化 |
| 可扩展 | 配置驱动 + Agent 适配器抽象 + LLM 结构化指令抽取 | 接新 Agent / 新指令均无需改核心代码 |

---

## 7. 交付里程碑（2–3 周）

| 周 | 目标 | 关键交付 |
|---|---|---|
| W1 | 底座 + 数据 | 配置层 + 指令 LLM 抽取 + Agent 适配器（builtin+http+offline）+ 现有 demo 数据跑通端到端 |
| W2 | 评测核心 | HybridSimulator + 红队 persona + L1/L2/L3 评分 + 多采样投票 + 证据引用 |
| W3 | 报告 + Demo 化 | Aggregator + HTML 报告 + Dashboard 扩展 + 对比模式 + README/演示脚本 |

**MVP 切线**（若时间压缩到 1 周）：去掉多采样投票（降为单次）、去掉 Dashboard 对比模式（用两次 HTML 报告手拼）、红队 persona 减到 1 类。

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| LLM 指令抽取在新指令上失败 | 上游污染下游 | Schema 严格校验 + 失败回环 + 生成 `*.review.md` 供人工核 |
| LLM Judge 不稳定 | 评分不可信 | L3 多采样投票 + 置信度告警；L1 硬约束兜底确定性 |
| 红队 persona 把 Agent 推到 OOD，反而失真 | 失败模式不真实 | 红队评分单独成桶，不污染普通 persona 总分 |
| 多 Agent 大规模采样调用成本高 | 跑不动 | 配置层支持 `filter_ids` / 抽样 / 并发；缓存 Judge 结果按 conversation_hash |
| Dashboard 实时推送复杂度 | 工程超时 | MVP 用轮询 + 进度文件；SSE 作为 stretch |

---

## 9. 开放问题（提交前需确认）

- 脱敏数据收到后，需快速跑通指令抽取并验证 schema 是否需要扩展
- 比赛是否对运行环境（Python 版本、是否允许联网调 LLM）有硬约束？影响 LLM Judge 是否能在评委机器上跑

---

（设计稿到此为止；实施细节进入实施计划阶段）
