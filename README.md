# Terraria Wiki RAG

把 [terraria.wiki.gg/zh](https://terraria.wiki.gg/zh) 全站爬下来做成 RAG 知识库。

## 技术栈

- **爬虫**: MediaWiki API (`/api.php`) + sitemap，限速 + 断点续传
- **清洗**: wikitext → 纯文本，保留属性表
- **切块**: 按 wiki section 切，长 section 二次切
- **Embedding**: BGE-M3（本地，中文强，支持稠密 + sparse 混合检索）
- **向量库**: Qdrant（本地文件持久化模式，无需 docker）
- **服务**: FastAPI

## 目录结构

```
terraria-rag/
├── pyproject.toml               # uv 管理
├── docker-compose.yml           # 可选，未来切到 server 模式时用
├── data/
│   ├── raw/                     # 原始 wikitext + metadata（jsonl）
│   ├── cleaned/                 # 清洗后的纯文本
│   └── qdrant/                  # 向量库本地存储
├── src/terraria_rag/
│   ├── config.py                # 全局配置（URL、限速、模型路径等）
│   ├── crawler/                 # 爬虫
│   ├── cleaning/                # wikitext 清洗
│   ├── chunking/                # 切块
│   ├── embedding/               # BGE-M3 封装
│   ├── store/                   # Qdrant 封装
│   ├── rag/                     # 检索 + （可选）reranker
│   └── api/                     # FastAPI 服务
└── scripts/                     # 一键执行脚本
    ├── 01_enumerate.py          # 列出所有页面标题
    ├── 02_crawl.py              # 拉取 wikitext
    ├── 03_clean_chunk.py        # 清洗 + 切块
    ├── 04_index.py              # embedding + 入库
    └── 05_serve.py              # 启动 API
```

## 快速开始

> 流水线（5 步）：`01_enumerate` → `02_crawl` → `03_clean_chunk` → `04_index` → `05_serve`
>
> 默认 API 端口 **8001**（在 `.env` 里改 `API_PORT`）。

### macOS / Linux

```bash
# 0. 复制 .env（按需改：HF 镜像、API 端口、embedding 设备等）
cp .env.example .env

# 1. 安装依赖（uv 会自动建 .venv 并装 Python 3.12）
uv sync --index-strategy unsafe-best-match

# 2. 下载 BGE-M3 模型（国内首选 ModelScope，~2.3GB，几分钟）
uv pip install modelscope
uv run modelscope download --model BAAI/bge-m3 --local_dir ./models/bge-m3
# 然后在 .env 把 EMBEDDING_MODEL 改成 ./models/bge-m3
# （海外用户可直接用默认的 EMBEDDING_MODEL=BAAI/bge-m3，会自动从 HF 下）

# 3. smoke test：先爬 10 个页面验证流程
uv run python scripts/01_enumerate.py --limit 10
uv run python scripts/02_crawl.py
uv run python scripts/03_clean_chunk.py
uv run python scripts/04_index.py --rebuild

# 4. 启动 API
uv run python scripts/05_serve.py
```

### Windows（PowerShell，推荐）

```powershell
# 0. 复制 .env（按需改）
Copy-Item .env.example .env

# 1. 安装依赖
uv sync --index-strategy unsafe-best-match

# 2. 下载 BGE-M3 模型（国内首选 ModelScope，~2.3GB）
uv pip install modelscope
uv run modelscope download --model BAAI/bge-m3 --local_dir .\models\bge-m3
# 然后在 .env 把 EMBEDDING_MODEL 改成 ./models/bge-m3
# （路径分隔符在 .env 里写 `/` 即可，跨平台兼容）

# 3. smoke test：先爬 10 个页面验证流程
uv run python scripts/01_enumerate.py --limit 10
uv run python scripts/02_crawl.py
uv run python scripts/03_clean_chunk.py
uv run python scripts/04_index.py --rebuild

# 4. 启动 API
uv run python scripts/05_serve.py
```

### Windows（CMD）

```bat
copy .env.example .env
uv sync --index-strategy unsafe-best-match
uv pip install modelscope
uv run modelscope download --model BAAI/bge-m3 --local_dir .\models\bge-m3

uv run python scripts\01_enumerate.py --limit 10
uv run python scripts\02_crawl.py
uv run python scripts\03_clean_chunk.py
uv run python scripts\04_index.py --rebuild
uv run python scripts\05_serve.py
```

> Windows 上若 `uv` / `python` 没在 PATH，先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)（`winget install --id=astral-sh.uv`）。

---

## 调用 API

启动后访问：

- **Swagger UI**：<http://localhost:8001/docs>
- **健康检查**：<http://localhost:8001/health>

### 查询接口示例

**bash / zsh / Git Bash（反斜杠续行）**

```bash
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"如何获得飞行员风镜？", "top_k":5}'
```

**Windows CMD（单行 — 不要用 `\` 续行，JSON 用双引号并转义内部 `\"`）**

```bat
curl -X POST http://localhost:8001/query -H "Content-Type: application/json" -d "{\"query\":\"如何获得飞行员风镜？\",\"top_k\":5}"
```

**PowerShell（推荐 `Invoke-RestMethod`，不用纠结引号）**

```powershell
Invoke-RestMethod -Uri http://localhost:8001/query -Method Post -ContentType "application/json" -Body '{"query":"如何获得飞行员风镜？","top_k":5}'
```

---

## 全量爬取

确认 smoke test 通过后（命令在三个平台一致，把 `--limit` 去掉即可）：

```bash
uv run python scripts/01_enumerate.py        # 不带 --limit
uv run python scripts/02_crawl.py            # 自动断点续传
uv run python scripts/03_clean_chunk.py
uv run python scripts/04_index.py --rebuild  # 重建向量库
```

预估（zh 站约 5000~8000 个页面）：

- 爬取：约 1.5~2.5 小时（限速 1 req/s，礼貌爬取）
- 切块 + 清洗：1~2 分钟
- Embedding 时间随设备（在 `.env` 里设 `EMBEDDING_DEVICE`）：
    - macOS（M 系列，`mps`）：约 10~20 分钟（实测 ~9 chunks/s）
    - Windows / Linux + NVIDIA GPU（`cuda`）：约 5~15 分钟
    - 纯 CPU（`cpu`）：1~3 小时，建议先用 `--limit` 试跑

## 已知 trade-off

- **术语保留英文**：terraria.wiki.gg 的 wikitext 大量使用 `{{tr|EnglishTerm}}`
  这种翻译模板，服务端渲染时才换成中文。我们直接用 wikitext 拿到的是 `Master Mode`、
  `Eye of Cthulhu` 这类英文。对 RAG 检索其实是好事（中英术语都能命中），但展示给最
  终用户时建议加一层后处理或在前端做术语字典映射。
- **没用 reranker**：BGE-M3 的稠密 + 稀疏混合检索 + RRF 已经够用。如果效果不够，
  可以再叠 `BAAI/bge-reranker-v2-m3` 做精排（已在依赖里）。
- **本地 Qdrant，单进程独占**：embedded 模式下同一时间只能有一个进程打开。要并
  发起服务就用 `docker compose up -d qdrant` 切到 server 模式。

## 礼貌爬取

- 默认 1 req/s，遵守 robots.txt
- User-Agent: `terraria-rag-bot/0.1 (personal study, https://github.com/yourname)`
- 用 MediaWiki API（比硬爬 HTML 友好得多）
- 失败自动重试 + 断点续传，避免重复打扰服务器
