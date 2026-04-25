# Dialog Agent Eval Demo

一个用于展示复杂外呼任务自动评测流程的 demo 项目。

## 包含内容

- 指令结构化
- 评测协议生成
- 用户模拟器
- Kimi API 驱动的 Agent / User Simulator
- 规则评分器
- 批量评测
- 前端展示页

## 主要目录

- `frontend/`: 展示页面
- `scripts/`: 数据处理与测试脚本
- `runner/`: 单场景与批量评测入口
- `outputs/`: 本地评测输出

## 环境变量

参考 `.env.example` 配置 Kimi API Key。

## 本地展示

```bash
python3 -m http.server 8000
```

打开：

`http://localhost:8000/frontend/index.html`
