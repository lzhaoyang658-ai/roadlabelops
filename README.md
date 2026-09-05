# RoadLabelOps

RoadLabelOps 是一个围绕 CVAT 构建的可恢复道路视频标注工作流。它把视频探测、场景切片、
模型预标注、人工验收、质量计算和 COCO/YOLO 数据集发布串成一条可追溯、不可静默覆盖的
流水线。

当前版本面向单用户或受信内网环境。工作流由确定性状态机驱动，并非自主 LLM Agent；
模型不能绕过人工审核直接成为最终标注。

## 产品能力

- 使用 FFmpeg/ffprobe 探测道路视频，并按固定时长生成可恢复的 Scene。
- 使用本地 Ultralytics YOLO 权重生成八类道路目标预标注。
- 创建 CVAT Project、Task 和 Job，并在人工验收后同步最终注释。
- 对预测和人工终稿计算 Precision、Recall、F1、无问题帧率及类别分布。
- 发布包含图片、COCO 注释、YOLO 标签、质量报告和 SHA-256 清单的数据集 Release。
- 在服务重启或操作失败后，从持久化 Journal 和状态记录恢复工作流。
- 通过 FastAPI、CLI 和响应式 Next.js 工作台提供同一套状态与安全边界。

八类标签为 `car`、`bus`、`truck`、`motorcycle`、`bicycle`、`pedestrian`、
`traffic_light` 和 `traffic_sign`。详细定义见[道路目标标注规范](docs/label-guide.md)。

## 架构

```text
Browser
   │
   ▼
Next.js workspace ──► FastAPI ──► WorkflowRuntime ──► LocalStore / Journal
                                      │
                                      ├──► FFmpeg / ffprobe
                                      ├──► local YOLO weights
                                      ├──► external CVAT service
                                      └──► immutable COCO / YOLO Release
```

主流程：

```text
video probe → scene split → CVAT task → auto label → human review
            → quality calculation → hash-bound manifest → verified release
```

后端状态是唯一事实来源。前端只展示并触发受控动作，不在浏览器中保存 CVAT 凭证或伪造
工作流状态。

## 环境要求

- Python 包支持 3.10+；`setup.sh` 使用经过验证的 Python 3.11 启动环境。
- [uv](https://docs.astral.sh/uv/)。
- Node.js 20.9+ 与 npm 10+。
- FFmpeg 和 ffprobe。
- 真实模式需要独立运行的 CVAT 2.74.x 和本地 `.pt` 模型权重。

macOS 可使用 `brew install ffmpeg` 安装 FFmpeg。CVAT、Docker、模型权重和输入视频均需
由使用者单独准备，不随本仓库分发。

## 快速开始

```bash
git clone https://github.com/lzhaoyang658-ai/roadlabelops.git
cd roadlabelops
./setup.sh
```

`setup.sh` 会创建权限为 `0600` 的本地 `.env`、安装锁定的 Python/前端依赖，并创建被
Git 忽略的运行目录。

先启动后端：

```bash
.venv/bin/roadlabelops doctor --demo-only
.venv/bin/uvicorn roadlabelops.api:app --reload --host 127.0.0.1 --port 8100
```

另开一个终端启动前端：

```bash
npm --prefix frontend run dev -- --port 3100
```

打开 <http://127.0.0.1:3100>。本地开发默认仅绑定回环地址；不要把后端 API 直接暴露到
公网。

## Demo 与真实模式

### Demo 模式

Demo 用确定性合成记录演示完整状态流转，不读取用户视频，不创建真实 CVAT 任务，也不调用
真实检测权重。运行后在工作台点击“使用演示数据”，或执行：

```bash
.venv/bin/roadlabelops demo
```

`doctor --demo-only` 通过只表示 Demo 运行面可用，不代表真实视频链路已经就绪。

### 真实模式

1. 在独立环境中启动 CVAT，并创建最小权限的账号或 Access Token。
2. 把经过来源和哈希核验的 YOLO `.pt` 文件放到仓库之外的受控目录。
3. 复制并填写 `.env`，至少配置 `CVAT_HOST`、一种 CVAT 凭证、
   `DETECTION_PROVIDER=yolo` 和 `DETECTION_MODEL` 的本地绝对路径。
4. 执行 `.venv/bin/roadlabelops doctor`；只有 CVAT、FFmpeg、可写目录和本地权重均通过
   检查时，真实链路才会显示就绪。
5. 在工作台上传 MP4、MOV 或 M4V，随后按界面提示完成建任务、预标注和人工审核。

推理过程不会隐式下载模型。已有人工标注时，自动标注刷新会默认拒绝覆盖。常用 CLI：

```bash
.venv/bin/roadlabelops list
.venv/bin/roadlabelops ingest /path/to/road-video.mp4 --scene-seconds 15
.venv/bin/roadlabelops run-to-review /path/to/road-video.mp4 --scene-seconds 15
.venv/bin/roadlabelops advance <session-id> <action> [--approve]
```

发布成功后，每个 Release 包含：

```text
annotations.coco.json
dataset.yaml
images/
labels/
quality.json
manifest.json
receipt.json
```

Release 目录不可覆盖；再次使用前可以重新计算文件哈希并验证完整性。

## 测试

后端：

```bash
uv sync --locked --extra dev --extra detection
uv run --frozen ruff check .
uv run --frozen pytest
```

前端：

```bash
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
```

浏览器流程变更还应运行：

```bash
npm --prefix frontend exec -- playwright install chromium
npm --prefix frontend run test:e2e
```

GitHub Actions 会在 Python 3.10/3.11 与项目固定的 Node.js 版本上执行对应检查。

## 部署

构建 standalone 前端并准备生产依赖：

```bash
make production-bootstrap \
  BACKEND_URL=http://127.0.0.1:8100 \
  NEXT_PUBLIC_CVAT_BASE_URL=https://cvat.example.com \
  NEXT_PUBLIC_SOURCE_URL=https://github.com/lzhaoyang658-ai/roadlabelops
```

部署修改版时，请把 `NEXT_PUBLIC_SOURCE_URL` 指向该版本对应的源码地址；页脚会向远程用户
展示这个入口。

通过真实就绪检查后分别启动两个服务：

```bash
make doctor ENV_FILE=/absolute/path/to/roadlabelops.env
make serve-backend ENV_FILE=/absolute/path/to/roadlabelops.env
make serve-frontend FRONTEND_HOST=127.0.0.1 FRONTEND_PORT=3100
```

生产配置、硬限额、探针、远程 CVAT 和反向代理边界见
[生产部署指南](docs/production-deployment.md)。当前 API 不提供多用户身份认证；远程访问必须
位于带身份认证的反向代理和网络 ACL 之后。

## 数据与安全边界

- 仓库不包含视频、模型权重、训练或评测数据集、CVAT 导出、生成的 Release、运行日志、
  凭证、内部实验记录或验收证据。
- `data/`、`runtime/`、`.env*`、模型权重和构建产物默认不会提交；贡献者仍需在提交前自行
  检查暂存区。
- 输入视频受文件大小、时长、分辨率、帧率、Scene 数量和切片总字节硬限额约束。
- 受管路径会执行边界检查；Release 和关键清单采用不可覆盖写入并绑定 SHA-256。
- `.env` 含凭证时在 POSIX 系统必须保持 `0600`，否则应用拒绝启动。
- CVAT 是独立外部服务；RoadLabelOps 不提供删除 CVAT Project/Task 或绕过人工审核的能力。
- 用户必须自行确认视频、模型和数据集的获取、处理与再分发权利。

如发现安全问题，请不要创建公开 Issue；按 [SECURITY.md](SECURITY.md) 中的方式进行私密报告。

## 贡献

开始贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。Pull Request 应保持单一目的、包含相应测试，
并且不得提交凭证、个人数据、模型权重或其他无权再分发的素材。

第三方依赖及资产边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

RoadLabelOps 公开署名为“未来。”。原创项目代码采用
[GNU Affero General Public License v3.0](LICENSE)（`AGPL-3.0-only`）。通过网络向用户
提供修改后的程序时，请特别注意 AGPL 对应的源代码提供义务。

第三方软件、字体、模型、视频和数据集继续适用各自的许可条款；项目许可证不会替代这些
条款。
