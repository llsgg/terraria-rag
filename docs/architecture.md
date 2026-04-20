# 系统架构

读这一份就能拿到全局图。后面每一份模块文档都默认你已经看过这份。

---

## 1. 我们在做什么

把整个 [terraria.wiki.gg/zh](https://terraria.wiki.gg/zh) 拉下来，做成一个**只读**的中文 Terraria 知识库，对外提供一个 `/query` 接口：输入一句自然语言问题，返回最相关的 wiki 片段（带页面链接、章节路径、原文）。

不接 LLM。LLM 该接 OpenAI、DeepSeek、本地 Ollama、还是通义/智谱，由调用方决定（见 [`api.md`](./api.md)）。我们只做**检索**这一层。

---

## 2. 设计原则（按重要性排序）

1. **每一步落盘，每一步可重跑**。流水线分 4 个独立脚本（见下），相邻两步通过文件而非内存交接。失败重跑只重跑那一步，不用从头来。
2. **断点续传 / 幂等**：再跑一次只补差集（爬虫按 `pageid.json` 是否存在判断；索引按 `--rebuild` 显式控制）。
3. **本地优先**：BGE-M3 本地推理，Qdrant embedded 模式（文件持久化，无 docker），整个项目可以离线跑。
4. **配置走 `.env`**：调速率、换站、换设备都改环境变量，不改代码。
5. **代码长期可读 > 一次性最优**：`mwparserfromhell`、`FlagEmbedding`、`qdrant-client` 这些都是各自领域的标准库，不重复造轮子。

---

## 3. 数据流（一图）

```
                       MediaWiki API
                       (api.php)
                            │
                            ▼
   ┌────────────────────────────────────────────────┐
   │  scripts/01_enumerate.py                       │
   │  └─ list=allpages → page_index.jsonl           │
   └────────────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────┐
   │  scripts/02_crawl.py                           │
   │  └─ prop=revisions (batch=50) → pages/*.json   │
   │     ✦ 限速 + 并发 + 断点续传                     │
   └────────────────────────────────────────────────┘
                            │
                            ▼  原始 wikitext + 元数据
   ┌────────────────────────────────────────────────┐
   │  scripts/03_clean_chunk.py                     │
   │  ├─ cleaning/wikitext.py                       │
   │  │   └─ wikitext → sections (纯文本 + 标题树)   │
   │  └─ chunking/splitter.py                       │
   │      └─ sections → chunks (≈512 tokens)        │
   │  → cleaned/chunks.jsonl                        │
   └────────────────────────────────────────────────┘
                            │
                            ▼  纯文本 chunks（带 section_path）
   ┌────────────────────────────────────────────────┐
   │  scripts/04_index.py                           │
   │  ├─ embedding/bge.py                           │
   │  │   └─ BGE-M3 → (dense 1024, sparse dict)     │
   │  └─ store/qdrant_store.py                      │
   │      └─ upsert(dense + sparse, payload)        │
   │  → data/qdrant/                                │
   └────────────────────────────────────────────────┘
                            │
                            ▼  本地 Qdrant collection
   ┌────────────────────────────────────────────────┐
   │  scripts/05_serve.py → api/server.py           │
   │  └─ POST /query                                │
   │     ├─ encode_query (BGE-M3)                   │
   │     └─ hybrid_search (Qdrant RRF fusion)       │
   └────────────────────────────────────────────────┘
                            │
                            ▼
                   { hits: [...top_k chunks...] }
```

---

## 4. 模块职责清单

| 目录 / 文件 | 职责 | 详细文档 |
|---|---|---|
| `crawler/api_client.py` | MediaWiki API 客户端：限速、重试、批量、重定向解析 | [`crawler.md`](./crawler.md) |
| `cleaning/wikitext.py` | wikitext → 纯文本，保留章节结构和 infobox 数据 | [`cleaning.md`](./cleaning.md) |
| `chunking/splitter.py` | sections → chunks，长 section 二次切 + overlap | [`chunking.md`](./chunking.md) |
| `embedding/bge.py` | BGE-M3 封装：同时产出 dense + sparse 向量 | [`embedding.md`](./embedding.md) |
| `store/qdrant_store.py` | Qdrant collection schema + upsert + hybrid 查询 | [`retrieval.md`](./retrieval.md) |
| `api/server.py` | FastAPI：`/query` + `/health` | [`api.md`](./api.md) |
| `config.py` | pydantic-settings，全部从 `.env` 加载 | [`operations.md`](./operations.md) |

---

## 5. 数据落盘 layout

```
data/
├── raw/
│   ├── page_index.jsonl       # 01 输出：所有 (pageid, title)
│   ├── pages/                 # 02 输出：每页一个 json
│   │   ├── 1234.json
│   │   └── ...
│   └── failures.jsonl         # 02 失败日志（如果有）
├── cleaned/
│   └── chunks.jsonl           # 03 输出：每个 chunk 一行
└── qdrant/                    # 04 输出 + 05 读取
    ├── collection/
    └── meta.json
```

为什么用 jsonl 而不是 parquet/sqlite：

- jsonl 可流式读、可 `head/tail` 看、可 `wc -l` 计数、可断行追加。我们的体量（~万级 pages、~5 万 chunks）完全在 jsonl 舒适区里。
- parquet 适合列式分析，我们这里没有列式查询需求。
- sqlite 会增加一个抽象层但带不来检索能力（检索归 Qdrant 干）。

---

## 6. 关键 trade-off 速查

| Trade-off | 我们的选择 | 代价 |
|---|---|---|
| 中文术语 vs 英文术语 | 不还原 `{{tr|EnglishTerm}}` 的中文映射，保留英文 | 检索更鲁棒，但展示给最终用户时英文术语会"漏出来"（前端可加术语字典补一刀） |
| 稠密 vs 稀疏 vs 二者皆要 | dense + sparse 混合 + RRF fusion | embedding 时多算一份 sparse、写盘大 ~30%，但召回质量明显更稳 |
| reranker 上 vs 不上 | 默认不上 | 简单。如果 top-1 不够准再加 `bge-reranker-v2-m3`（已在依赖里） |
| Qdrant embedded vs server | embedded（文件模式） | 同时只能一个进程打开。要并发起服务/索引，切到 `docker compose up -d qdrant` |
| 切块按 token vs 按字符 | 按字符（中文 1.6 字符≈1 token 估） | 不依赖 tokenizer，BGE-M3 内部 `max_length=1024` 兜底真的超长会截断 |
| 异步 vs 线程池 | 线程池（`ThreadPoolExecutor`） | 对几千页量级，asyncio 收益极小、复杂度高 |

---

## 7. 不在范围内（intentionally out of scope）

- **多模态**：不抓图片、视频。如果哪天要做"看图识装备"，需要单独搭一条 CLIP 流水线。
- **增量爬取（diff crawl）**：目前是"按 pageid 是否落盘"的粗粒度增量。要做"自上次以来更新的页面"，需要存 `revid` 并用 `prop=info` 比对（见 [`operations.md`](./operations.md) 的"未来工作"小节）。
- **多语言**：只爬 `WIKI_LANG=zh`。改一行就能换 en，但 chunk 大小、infobox 模板名等参数没在英文上调过。
- **写接口 / 编辑 wiki**：只读知识库，不会发回任何修改。
- **LLM 接入**：刻意不做。`/query` 返回 chunks，由调用方决定接哪个 LLM、怎么 prompt。

---

## 8. 扩展点

如果你想做下面这些事，对应改这里：

| 想做的事 | 主要改动点 |
|---|---|
| 换一个 MediaWiki 站（如 minecraft.wiki） | `.env` 改 `WIKI_BASE_URL`，可能还要调 `cleaning/wikitext.py` 里的 infobox 模板名 |
| 加术语词典（英→中） | 在 `chunking/splitter.py` 输出前过一遍，或在 `api/server.py` 输出前过一遍 |
| 加 reranker | `api/server.py` 在 `hybrid_search` 之后插入 `bge-reranker-v2-m3` |
| 换 embedding 模型 | `embedding/bge.py` 改成新模型；改 `EMBEDDING_MODEL` 环境变量；**记得 `04_index.py --rebuild`** |
| 切到 Qdrant server 模式 | `store/qdrant_store.py` 的 `QdrantClient(path=...)` 改成 `QdrantClient(url=...)` |
| 加 LLM 综合答案 | 新写一个 `/answer` 路由，里面调 `/query` 拿 chunks，再喂给 LLM。**保留 `/query` 不变** |
| 加分面过滤（如"只搜武器"） | upsert 时把 `categories` 写进 payload，`hybrid_search` 加 `filter=qm.Filter(...)` |

---

下一步建议读：

- 想理解爬取细节 → [`crawler.md`](./crawler.md)
- 想理解为什么这么切块 → [`chunking.md`](./chunking.md)
- 想理解为什么混合检索 → [`retrieval.md`](./retrieval.md)
- 想部署 / 排查问题 → [`operations.md`](./operations.md)
