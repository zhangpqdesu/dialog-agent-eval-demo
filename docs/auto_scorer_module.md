# 自动评分器模块

本模块负责对一段对话执行结果进行自动评分，并产出结构化评测报告。

## 文件

- 评分引擎：
  `evaluator/auto_scorer.py`
- 端到端 runner：
  `runner/run_evaluation.py`

## 当前输入

- 评测版指令数据：
  `data/processed/dialog_instruction_eval_examples.json`
- 用户模拟器场景：
  `data/processed/user_simulator_scenarios.json`

## 当前输出

- 单次评测报告 JSON
- 包含：
  - `conversation`
  - `report.overall_score`
  - `report.category_scores`
  - `report.success_results`
  - `report.failure_results`

## 当前评分方式

- `success_criteria`
  - 任务目标命中
  - 开场命中
  - 流程步骤覆盖
  - FAQ 相关知识命中
  - 单轮长度约束
- `failure_conditions`
  - 漏任务目标
  - 漏流程
  - 超字数
  - 超范围问题处理失败
  - 特殊结束场景处理失败

## 当前边界

- 这是规则评分器，不是 LLM Judge。
- 适合先验证评测流程、指标设计和报告结构。
- 后续可以把打分逻辑替换为：
  - 规则评分
  - LLM 语义评分
  - 混合评分

## 运行

运行单次演示：

```bash
'/Users/zhangpq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' runner/run_evaluation.py
```

当前默认配置：

- `Agent = kimi-k2-turbo-preview`
- `UserSimulator = kimi-k2-turbo-preview`
- `Judge = 规则`
