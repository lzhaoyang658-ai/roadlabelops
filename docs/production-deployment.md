# RoadLabelOps 生产启动与 CVAT 2.74 配置

RoadLabelOps V1 是单用户工作流。生产就绪的含义是：数据目录可写、
FFmpeg/ffprobe 可用、CVAT 已通过认证健康检查，且选定的真实检测器和
本地权重可用。`mock` 只表示 Demo 就绪，不代表真实视频全流程就绪。

> **网络边界：** V1 API 本身不提供多用户身份认证。后端应保持绑定
> `127.0.0.1` 或受信网络；如需远程访问，必须在前面配置身份认证反向代理
> 和网络 ACL。不要把 API 端口直接暴露到公网。

## 1. 固定版本与干净安装

已验证的基线是 Python 3.11、Node.js 20.9 或更高版本、npm 10+、FFmpeg，
以及 CVAT Server / SDK 2.74.x。在项目根目录执行：

```bash
make production-bootstrap \
  BACKEND_URL=http://127.0.0.1:8100 \
  NEXT_PUBLIC_CVAT_BASE_URL=http://localhost:8080 \
  NEXT_PUBLIC_SOURCE_URL=https://github.com/lzhaoyang658-ai/roadlabelops
```

`setup.sh` 会检查 Python/Node/FFmpeg，通过 `uv sync --frozen` 安装 Python
锁定依赖，并通过 `npm ci` 安装前端锁定依赖。Make 目标随后构建
Next.js standalone 产物。

`BACKEND_URL` 是 Next.js 服务端代理的构建时变量，
`NEXT_PUBLIC_CVAT_BASE_URL` 是浏览器打开 CVAT 的构建时变量，
`NEXT_PUBLIC_SOURCE_URL` 是页脚展示的对应源码地址。部署修改版时必须把源码地址改为
该版本实际可获取的位置。任一值改变后都必须重新执行 `make build-frontend ...`，不能只
重启 standalone 进程。

## 2. 工作目录和环境文件合约

- 默认环境文件固定为项目根目录的 `.env`，不取决于启动命令的当前目录。
- 要使用其他文件，在进程环境中设置绝对路径
  `ROADLABELOPS_ENV_FILE=/etc/roadlabelops/production.env`。相对路径会被拒绝。
- `.env` 中的相对 `ROADLABELOPS_DATA_DIR`、`ROADLABELOPS_RUNTIME_DIR` 和
  `DETECTION_MODEL` 都相对
  `ROADLABELOPS_PROJECT_DIR` 解析，不相对进程的当前目录解析。
- 生产环境建议三个路径全部使用绝对路径。含 CVAT 凭证的环境文件
  必须设为 `0600`（例如 `chmod 600 /etc/roadlabelops/production.env`），
  否则应用会拒绝启动。

从 `.env.example` 复制后，至少确认：

```dotenv
ROADLABELOPS_ENV=production
ROADLABELOPS_PROJECT_DIR=/srv/roadlabelops
ROADLABELOPS_DATA_DIR=/srv/roadlabelops-data
ROADLABELOPS_RUNTIME_DIR=/srv/roadlabelops-runtime
ROADLABELOPS_FINAL_HOLDOUT_TASK_IDS=<final-holdout-task-id>
ROADLABELOPS_FINAL_HOLDOUT_JOB_IDS=<final-holdout-job-id>
CVAT_HOST=https://cvat.example.com
CVAT_ACCESS_TOKEN=<secret>
DETECTION_PROVIDER=yolo
DETECTION_MODEL=/srv/roadlabelops-models/yolo11n.pt
```

最终留出集没有内置 Task/Job 编号。运行最终留出集构建或评测工具时，必须通过
上面两个变量配置逗号分隔的正整数 ID，或向相应命令显式传入 ID；配置为空、
格式非法或与显式参数不一致都会失败。训练与恢复脚本无论是否配置 ID，始终拒绝
`data/holdout` 和带 `final-holdout` 语义的路径；配置 ID 后还会拒绝对应 Task/Job
路径，防止留出集进入训练输入。

视频导入必须同时保留软提示与不可绕过的安全上限，建议先使用默认值：

```dotenv
ROADLABELOPS_MAX_VIDEO_DURATION_SECONDS=300
ROADLABELOPS_ABSOLUTE_MAX_VIDEO_DURATION_SECONDS=7200
ROADLABELOPS_MAX_SCENE_COUNT=720
ROADLABELOPS_MAX_SPLIT_OUTPUT_BYTES=8589934592
```

`MAX_VIDEO_DURATION_SECONDS` 只控制需要人工确认的阈值；其他三项是即使确认
长视频也不能绕过的硬限。切片前还会按源文件大小、分辨率、帧率和时长
估算存储需求，切片中累计输出字节；超限时会清理 staging，不发布半成品。

首次启动前由运行用户创建受管目录，例如：

```bash
install -d -m 750 /srv/roadlabelops-data /srv/roadlabelops-runtime
```

## 3. 显式准备 YOLO 权重

推理过程不会自动下载权重。首选做法是把已核验的 `.pt` 文件复制到
受控模型目录，并在 `DETECTION_MODEL` 中配置其绝对路径。

如果确实要由 Ultralytics 在可联网的安装机上获取官方基础权重，必须显式
执行一次：

```bash
cd /srv/roadlabelops
.venv/bin/python -c 'from ultralytics import YOLO; YOLO("yolo11n.pt")'
shasum -a 256 yolo11n.pt
```

将校验值录入运维记录后再部署。`doctor` 和 readiness 会检查本地文件确实是
结构完整的 PyTorch checkpoint archive；权重缺失、为空或损坏都会显式失败，
真实推理也不会在请求期间触发网络下载。

## 4. CVAT 2.74 本地与远程配置

### 本地 Docker

在 RoadLabelOps 目录之外固定 CVAT 2.74.0 源码标签：

```bash
git clone --depth 1 --branch v2.74.0 https://github.com/cvat-ai/cvat.git ../cvat-2.74
cd ../cvat-2.74
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

在浏览器打开 `http://localhost:8080`，确认能登录；然后在 RoadLabelOps
`.env` 中填入 `CVAT_HOST` 以及用户名/密码或 Access Token。不要同时把
凭证写入前端环境文件。

### 远程 CVAT

远程服务应提供 HTTPS，并保证 RoadLabelOps 后端可访问 API、浏览器可访问
CVAT 页面。`CVAT_HOST` 使用后端可达地址；构建前端时的
`NEXT_PUBLIC_CVAT_BASE_URL` 使用浏览器可达地址。当服务端不是 2.74.x
时，先在预发环境完成 SDK 兼容验收，不要直接替换生产端。

## 5. 就绪检查与启动

```bash
make doctor ENV_FILE=/etc/roadlabelops/production.env
make serve-backend ENV_FILE=/etc/roadlabelops/production.env
make serve-frontend FRONTEND_HOST=127.0.0.1 FRONTEND_PORT=3100
```

`roadlabelops doctor` 只有真实全流程就绪时才退出 `0`；缺少 CVAT、FFmpeg
或真实检测器时退出 `2`。只验证 Demo 运行面可执行
`roadlabelops doctor --demo-only`。

运行时探针：

```bash
curl -fsS http://127.0.0.1:8100/api/v1/health/live
curl -fsS http://127.0.0.1:8100/api/v1/health/ready
```

`/health/live` 只表示 API 进程存活，不访问任何下游。`/health/ready` 每次都实际
测试数据目录写入、FFmpeg/ffprobe、CVAT 认证健康与 2.74.x 兼容性，以及本地
检测器 checkpoint 完整性；未就绪时
返回 HTTP 503。旧的 `/api/v1/health` 保留为 liveness 别名。
