# 参与贡献

感谢你改进 RoadLabelOps。提交代码前，请先搜索现有 Issue 和 Pull Request，避免重复工作。较大的功能、数据模型变更或兼容性改动，建议先创建 Issue 说明使用场景、边界和验收方式。

## 开发环境

需要准备：

- Python 3.10 或 3.11；项目验证环境默认使用 3.11。
- [uv](https://docs.astral.sh/uv/)。
- Node.js（版本见 `.nvmrc`）和 npm。
- FFmpeg 与 ffprobe。仅运行单元测试通常不需要启动 CVAT；验证真实工作流时需要兼容的 CVAT 2.74.x。

安装依赖：

```bash
uv sync --locked --extra dev --extra detection
npm --prefix frontend ci
```

需要运行本地应用时，复制示例配置并只在本机填写凭证：

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

不要提交 `.env`、访问令牌、CVAT 凭证、模型权重、用户视频、生成的数据集或包含本机绝对路径的证据文件。

## 本地检查

后端：

```bash
uv run --frozen ruff check .
uv run --frozen pytest
```

前端：

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
```

浏览器流程发生变化时，再运行端到端测试；首次使用需要安装 Playwright 浏览器：

```bash
npm --prefix frontend exec -- playwright install chromium
npm --prefix frontend run test:e2e
```

## 变更原则

- 保持工作流可恢复、可审计和默认拒绝越权操作。
- 不降低路径边界、哈希校验、不可覆盖 Release 或人工审核门禁。
- 新增行为应有自动化测试；修复缺陷时优先加入能复现问题的回归测试。
- UI 变更需同时检查桌面端和窄屏，不让动画阻碍操作或可访问性。
- 不在测试中依赖外网、真实 CVAT 凭证或未纳入仓库的本地文件。
- 文档中的命令、配置项和产品能力应与实现同步。

## Pull Request

Pull Request 请保持单一目的，并包含：

- 问题背景和解决方案。
- 影响范围、兼容性及安全考虑。
- 已运行的测试与结果。
- UI 变更的截图或短视频（如适用）。
- 尚未解决的限制或后续事项。

提交前请确认 CI 全部通过。维护者可能要求拆分过大的变更或补充测试。本项目已采用
[AGPL-3.0-only](LICENSE)；提交贡献即表示你有权按该许可证授权相应代码。第三方内容必须
同时保留其自身要求的许可和归属信息。

安全漏洞不要放在公开 Issue 中；请按 [安全策略](SECURITY.md) 私密报告。
