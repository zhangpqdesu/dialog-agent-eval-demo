# 用户模拟器模块

本模块基于评测版指令数据，生成可执行的用户模拟场景，并提供首版状态机引擎。

## 输入

- 评测版指令数据：
  `data/processed/dialog_instruction_eval_examples.json`

## 产出文件

- 用户模拟器场景 schema：
  `data/schemas/user_simulator.schema.json`
- 用户模拟器场景数据：
  `data/processed/user_simulator_scenarios.json`
- 状态机引擎：
  `simulator/user_simulator.py`
- 场景生成脚本：
  `scripts/build_user_simulator.py`

## 场景结构

每个场景包含：

- `scenario_id`
- `instruction_id`
- `profile_id`
- `initial_user_utterance`
- `initial_state_id`
- `states`
- `max_turns`

每个状态包含：

- `state_id`
- `intent`
- `expected_agent_signals`
- `default_response`
- `fallback_response`
- `transition_to`
- `terminal`

## 运行方式

先生成场景：

```bash
'/Users/zhangpq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' scripts/build_user_simulator.py
```

再运行演示：

```bash
'/Users/zhangpq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' simulator/user_simulator.py
```

## 当前行为

- 根据 `user_simulator_profiles` 自动生成多个测试场景。
- 每个场景绑定一个首版状态机。
- 状态机会根据模型上一轮回复中的关键词，决定当前用户使用默认回应还是回退回应。
- 若达到终止状态或超过 `max_turns`，会结束会话。

## 当前适用范围

- 首版适合做自动化回归测试和流程覆盖测试。
- 适合快速验证模型在不同用户画像下的稳定性。
- 目前仍是规则状态机，不是 LLM 驱动的高拟真用户代理。

## 下一步可扩展

- 加入基于评分项的动态状态跳转。
- 加入槽位填充与变量实例化。
- 加入多候选回复采样，提升对话多样性。
- 将用户模拟器输出直接接入自动评分流水线。
