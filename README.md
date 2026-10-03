# FinCopilot

财报智能问答与分析服务 —— 基于 A 股上市公司年报的 RAG + Agent 系统。

技术难点锚定在**中文年报的表格数值问答**：量纲混用（元/万元/亿元）、科目口径不一
（营业收入 ≠ 营业总收入）、表格跨页与合并单元格。

完整架构设计见 [`../fincopilot-architecture.html`](../fincopilot-architecture.html)。

---

## 当前进度

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0** | 骨架与地基：配置体系、Provider Registry、三存储连接、健康探针、Langfuse | ✅ 完成 |
| **M1** | 离线入库链路：解析 → 分块 → 表格元数据 → 向量化 → 双写 | ✅ 完成 |
| **M2** | 在线 RAG 与首个可演示版本 | ✅ 完成 |
| M3 | 评估体系与基线（320 条评估集） | ⬜ |
| M4 | 检索优化消融（顺序 / 交叉 / 逆向三阶段） | ⬜ |
| M5 | Agent 能力（工具、护栏、Checkpoint、HITL） | ⬜ |
| M6 | 成本优化与交付包装 | ⬜ |

---

## 快速开始

### 1. 起本地依赖

```bash
podman compose up -d
```

拉起 PostgreSQL、Redis、Milvus（含 etcd + minio）。Milvus 首次启动约需 1–2 分钟。

本项目使用 **Podman**（rootless，WSL2 后端），已在 Podman 4.8 +
Docker Compose v5.5.1 上验证通过。Podman 暴露 Docker 兼容的
API 端点（`npipe:////./pipe/docker_engine`），标准 compose 工具可直接使用，
且不需要管理员权限。

> 架构硬约束：整套系统必须能在本机离线跑通（模型调用除外）。做不到这点，
> 云上一出问题就只能干等。

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

### 3. 配置

```bash
copy .env.example .env        # Windows
```

存储的默认值对应 compose 配置，可直接用。模型 Key 可暂时留空——
不填也能完成存储连通性验收，只是跳过模型调用。

百炼 Key 获取：[百炼控制台](https://bailian.console.aliyun.com) → 右上角头像 →
API-KEY 管理 → 创建。环境变量名沿用 `DASHSCOPE_API_KEY`（百炼前身为灵积 DashScope）。

### 4. 验收

```bash
python -m scripts.smoke        # 配置 + Provider + 三存储（不花钱）
python -m scripts.smoke --llm  # 额外调一次模型（需已填 Key）
pytest -q                      # 17 个测试，不依赖外部服务
```

### 5. 启动服务

```bash
uvicorn app.main:app --reload
```

- http://127.0.0.1:8000/healthz — 进程存活
- http://127.0.0.1:8000/readyz — 存储连通性 + Key 配置状态 + 当前实验
- http://127.0.0.1:8000/docs — 接口文档

### 6. 准备语料并入库

```bash
python -m scripts.fetch_reports            # 从巨潮下载年报（含内容校验）
python -m alembic upgrade head             # 建表
python -m scripts.init_stores              # 建 Milvus collection
python -m scripts.ingest --pages 20        # 入库，限页数以控制 embedding 成本
```

也可以走接口异步入库，此时需要另开一个 Worker 进程：

```bash
python -m arq app.worker.Settings          # Worker（与 API 同一镜像，不同启动命令）
```

```
POST   /api/v1/documents                上传 PDF，立即返回 job_id
GET    /api/v1/documents                文档列表与入库状态
GET    /api/v1/documents/jobs/{job_id}  任务进度
POST   /api/v1/documents/{id}/reindex   按指定实验配置重建索引
DELETE /api/v1/documents/{id}           删除文档及其向量
```

> 同一文档可在不同实验配置下并存入库：块按 `strategy` 字段隔离，
> 这是 M4 消融实验能横向对比的前提。

### 7. 开始提问

```bash
uvicorn app.main:app                           # 后端
chainlit run frontend/app.py --port 8001       # 前端（另开一个终端）
```

打开 http://127.0.0.1:8001 即可对话，回答附带页码级引用。

也可以直接调接口：

```
POST /api/v1/chat          非流式，供评估脚本使用
POST /api/v1/chat/stream   流式，事件：meta / route / step / token / citation / done
```

实测表现（exp02_heading，两份年报约 1500 块）：

| 指标 | 实测 |
|---|---|
| 单次问答成本 | ¥0.0026 – 0.0033 |
| 首字延迟 | 1.4 – 2.8 s |
| 完整回答 | 1.7 – 3.3 s |

示例回答：

> 贵州茅台2025年第一季度营业收入为 50,600,957,885.78 元 [4]
> 宁德时代2025年研发投入为 22,146,581 千元（即 221.47 亿元），该金额为费用化研发支出 [2][4]

两家公司量纲不同（元 / 千元），均被正确声明并换算——这是量纲处理链路的最终兑现。

---

## 环境踩坑记录

搭建 M0 时实际遇到并解决的问题，换机器重建时可直接参考。

| 现象 | 原因 | 解法 |
|---|---|---|
| Redis 连接超时，但容器健康、端口可达 | Windows 上 `localhost` 优先解析到 IPv6 `::1`，而容器端口转发只监听 IPv4 | 连接串一律用 `127.0.0.1`，不用 `localhost` |
| `minio/minio` 镜像拉取报 `denied` | 该镜像的 RELEASE tag 已从 Docker Hub 下架（不是网络问题，用 `hello-world` 可验证网络正常） | 改用 Milvus 官方镜像的副本 `milvusdb/minio` |
| Milvus 起不来、连不上 etcd | etcd 的 `advertise-client-urls` 写成了 `127.0.0.1`，容器间访问必须用服务名 | 改为 `http://etcd:2379` |
| MinIO 启动失败，9001 端口被占 | 9001 落在 Windows 系统保留端口范围内（占用进程为 System PID 4） | 不暴露 9001（Milvus 走容器网络访问 minio:9000，宿主机不需要） |
| 数据库容器权限错误 | Podman rootless 在 WSL 中对 Windows 目录无写权限 | 用命名卷而非绑定挂载到 `./volumes/` |
| 全局 `python` 指向 Store 占位符 | Windows 的应用执行别名拦截 | 用 venv 内的解释器，或关闭设置里的 python.exe 别名 |

另：实际安装的依赖版本显著高于初版约束（langchain 1.x、langgraph 1.x、
langfuse 4.x、pymilvus 3.x），均已验证兼容并收紧了版本范围。
**Milvus 服务端也已进入 3.x**，M1 建 collection 时需确认 BM25 Function 等
特性在 3.x 下的写法。

---

## 项目结构

```
app/
├── config/        配置体系（环境变量 + YAML + 实验 profile）
├── providers/     Provider Registry —— 所有模型调用的唯一入口
├── store/         PostgreSQL / Redis / Milvus 连接管理
├── observability/ Langfuse trace + 成本核算
├── graph/         LangGraph 主图与状态定义
├── api/           FastAPI 路由
└── schemas/       请求响应模型

configs/
├── settings.yaml          非敏感运行时配置
├── models.yaml            Provider 配置表与价格
└── experiments/*.yaml     每组消融实验一个 profile
```

---

## 设计约定

几条在后续开发中不应被打破的规则，每条都对应架构文档中的一条决策记录：

| 约定 | 出处 |
|---|---|
| Provider 层是配置表，不是类继承 —— 加厂商只改 YAML | ADR-004 |
| 语义缓存必须叠加实体硬约束，不可只靠向量相似度 | ADR-013 |
| 主指标是端到端答案正确率，RAGAS 只作归因辅助 | ADR-014 |
| 所有可对比的技术选择必须是配置项，不是代码分支 | ADR-010 |
| 检索失败时拒答，不允许模型脱离文档自由发挥 | ADR-009 |
| 依赖必须锁定版本 —— 否则评估结果无法复现 | §13.4 |

---

## 开发纪律

1. 每个模块完成后，合上代码复述一遍设计意图。讲不顺就是没懂，回去补。
2. 每个里程碑记录 2–3 个 bad case：现象 → trace 定位 → 根因 → 方案 → 效果。
3. 有技术取舍就补一条 ADR，半页即可。

这三件事持续做，收尾时技术叙事的素材自然齐备；事后补写的记录经不起追问。
