# 指令结构化模块

本模块用于将 Excel 中的复杂外呼任务指令，转换为后续评测系统可直接消费的结构化 JSON 数据。

## 输入

- 原始 Excel：
  `/Users/zhangpq/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_jeyv1v60ynqc22_6797/msg/file/2026-04/命题二：外呼任务对话模型指令示例.xlsx`

## 产出文件

- 基础 schema：
  `data/schemas/dialog_instruction.schema.json`
- 基础标准化样例：
  `data/processed/dialog_instruction_examples.json`
- 评测专用 schema：
  `data/schemas/dialog_instruction_eval.schema.json`
- 评测专用样例：
  `data/processed/dialog_instruction_eval_examples.json`

## 基础版用途

基础版保留任务原貌，统一抽取：

- `role`
- `task`
- `opening_line`
- `call_flow`
- `knowledge_points`
- `constraints`

适合做指令清洗、数据管理和后续人工检查。

## 评测版用途

评测版在基础版之上补充：

- `instruction_core.required_information`
- `success_criteria`
- `failure_conditions`
- `user_simulator_profiles`

适合直接给：

- 用户模拟器
- 自动评分器
- 对话执行日志分析器

## 生成方式

运行：

```bash
'/Users/zhangpq/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' scripts/extract_instruction_schema.py
```

## 当前规则

- 使用标题切分原始 Markdown 风格指令块。
- 自动抽取显式约束，如字数限制、禁用表达、开车场景挂断、超范围兜底回复。
- 根据流程步骤、知识点和约束，自动生成首版成功标准与失败条件。
- 根据任务域和约束，自动生成首版用户模拟画像。

## 当前边界

- `success_criteria` 和 `failure_conditions` 目前是规则生成的首版，可继续人工精修。
- `user_simulator_profiles` 是通用模板加领域特化，不是最终的状态机脚本。
- 若后续 Excel 格式新增字段，需要同步扩展提取规则。
