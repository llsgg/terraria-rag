# API 设计（FastAPI）

把 [`embedding.md`](./embedding.md) + [`retrieval.md`](./retrieval.md) 通过一个最小的 HTTP 接口暴露出去。

代码：`src/terraria_rag/api/server.py`，启动入口 `scripts/05_serve.py`。

---

## 1. 设计哲学：只做检索，不接 LLM

`/query` 返回最相关的 wiki chunks，**不做生成**。原因：

| 不做 LLM 的理由 | 解释 |
|---|---|
| LLM 选择高度个人化 | OpenAI / DeepSeek / Claude / 通义 / 智谱 / 本地 Ollama / vLLM……各家 SDK、计费、prompt 风格都不一样。硬绑一家就排斥所有其他用户 |
| LLM 不该绑定语言 | 调 LLM 是 IO 密集，FastAPI 当然能做，但很多人会想接到自己的 Node / Go / Java 后端里 |
| 检索质量 vs 生成质量正交 | 把它们解耦后可以独立迭代 |
| 权限 / 审计 | 谁调用 LLM、多少 token、走哪个 key —— 留给上层处理更干净 |
| 测试更容易 | 检索接口可以纯本地端到端测，不需要 mock 任何在线服务 |

如果以后要加 LLM，**新加路由 `/answer`**，里面调 `/query` 拿 chunks 再喂给 LLM。**`/query` 永远保持纯检索语义。**

---

## 2. 接口

### `POST /query`

**Request**

```json
{
  "query": "如何获得飞行员风镜？",
  "top_k": 5
}
```

| 字段 | 类型 | 默认 | 约束 |
|---|---|---|---|
| `query` | str | 必填 | 1 ≤ len ≤ 2000 |
| `top_k` | int? | `RETRIEVAL_TOP_K`（默认 8） | 1 ≤ x ≤ 50 |

**Response**

```json
{
  "query": "如何获得飞行员风镜？",
  "total_chunks": 50213,
  "hits": [
    {
      "score": 0.0312,
      "title": "飞行员风镜",
      "section_path": "飞行员风镜 > 获取",
      "url": "https://terraria.wiki.gg/zh/wiki/飞行员风镜",
      "text": "# 飞行员风镜 > 获取\n\n秃鹰在沙漠..."
    }
  ]
}
```

| 字段 | 含义 |
|---|---|
| `query` | 原样回显，方便调试 |
| `total_chunks` | collection 里总共有多少 chunks，用来 sanity check |
| `hits` | 按 RRF 分数降序排，最大 `top_k` 条 |
| `hits[].score` | RRF 分数（**只在同一次 response 内可比**，不要做阈值过滤，见 [`retrieval.md`](./retrieval.md)） |
| `hits[].text` | 已经包含 section 前缀，直接展示给用户或喂给 LLM 都行 |
| `hits[].url` | 直接打开就是 wiki 原页，引用更可信 |

### `GET /health`

```json
{ "status": "ok", "indexed_chunks": 50213 }
```

用于：

- 启动后探活（k8s liveness / docker healthcheck）。
- 部署完确认 `indexed_chunks > 0`，没 0 那就是模型加载完但 collection 是空的（典型："忘了跑 04_index.py"）。

---

## 3. 模型在内存里的生命周期

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    print("[boot] Loading BGE-M3 ...")
    _state["embedder"] = BGEM3Embedder()       # ~10s, ~2.3GB RAM
    print("[boot] Connecting Qdrant ...")
    _state["store"] = QdrantStore()
    print("[boot] Ready. Indexed chunks:", _state["store"].count())
    yield
    _state["store"].close()
```

关键点：

- **模型只加载一次**（`lifespan` 是 FastAPI 0.93+ 推荐的启动钩子）。每次 `/query` 复用同一个 `BGEM3Embedder` 实例。否则每次请求都加载 ~2.3GB 模型，10 秒 + 一次 OOM。
- 用 dict `_state` 而不是模块级 `embedder = None` —— `lifespan` 闭包写起来更干净，也方便测试时替换 mock。
- `store.close()` 在退出时调一下，让 Qdrant embedded 模式优雅关掉文件句柄（不调也行，但留个好习惯）。
- `reload=False`（见 `05_serve.py`）：dev 时如果开 `reload=True`，每次代码改动都重新加载模型，要等 ~15 秒。除非你确实在改 server 代码，否则别开。

---

## 4. 错误处理

```python
try:
    hits: list[RetrievedChunk] = store.hybrid_search(q_emb.dense, q_emb.sparse, top_k)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"search failed: {e}") from e
```

只兜底了一处：search 失败转 500。其他异常路径：

| 场景 | FastAPI 默认行为 |
|---|---|
| `query` 太长 / 太短 | pydantic 422，附校验信息 |
| `top_k` 越界 | pydantic 422 |
| BGE-M3 OOM（非常长输入） | 500 |
| Qdrant 文件被另一个进程占用 | 启动时 `lifespan` 直接抛，server 起不来（这是你想要的，**别让一个半启动的 server 跑着**）|

刻意没做：

- 没加 `Retry-After` —— 这是只读服务，速率控制留给前置 nginx / API gateway。
- 没加 `X-Request-ID` —— 单进程小项目用不上。生产里加一行 middleware 就行。
- 没加 streaming —— 检索结果就那么大，一次性 JSON 比 SSE 简单。

---

## 5. 部署形态

### 5.1 单机直起（dev / 个人用）

```bash
uv run python scripts/05_serve.py
```

`uvicorn` 单 worker，监听 `API_HOST:API_PORT`（默认 `0.0.0.0:8001`）。**所有模型都在这一个进程里**，~3GB RAM。

### 5.2 多 worker（小生产）

⚠️ **不要直接 `uvicorn --workers 4`**：

- BGE-M3 模型 ~2.3GB × 4 worker = 9.2GB RAM；
- Qdrant embedded 模式**同一时间只允许一个进程打开 `data/qdrant/`** —— 多 worker 直接互相锁死。

要做多 worker：

1. **Qdrant 切到 server 模式**：

   ```bash
   docker compose up -d qdrant   # 项目里有 docker-compose.yml 占位
   ```

   然后改 `qdrant_store.py`：`QdrantClient(url="http://localhost:6333")`。
2. **embedder 仍然在每个 worker 里加载**（PyTorch 模型不易跨进程共享内存）；接受 RAM 翻倍代价。
3. 如果想省 embedder 内存，独立部署一个 embedding microservice（`POST /embed`），server 侧改成走它。

### 5.3 反代 + HTTPS

加 nginx / caddy 反代到 `127.0.0.1:8001`。FastAPI 这一层不做 HTTPS / 限流 / 鉴权，留给反代。

---

## 6. 一个完整调用例子

```python
import httpx

r = httpx.post("http://localhost:8001/query", json={
    "query": "克苏鲁之眼怎么打？",
    "top_k": 5,
}, timeout=30)
hits = r.json()["hits"]

for h in hits:
    print(f"[{h['score']:.4f}] {h['section_path']}")
    print(h["url"])
    print(h["text"][:200], "...\n")
```

如果要拿这些 chunks 喂给 LLM 生成答案：

```python
context = "\n\n---\n\n".join(
    f"来源：{h['url']}\n章节：{h['section_path']}\n\n{h['text']}"
    for h in hits
)

prompt = f"""根据下面的 Terraria Wiki 资料，回答用户问题。
要求：
- 只用资料里的信息回答，没写的就说"资料里没有"。
- 在回答末尾列出引用的 url。

资料：
{context}

用户问题：{query}
"""

# 然后调你想用的 LLM……
```

---

## 7. 不采用的方案

| 方案 | 为什么没用 |
|---|---|
| 在 `/query` 里直接调 LLM 出答案 | 见第 1 节。LLM 选择是用户的事 |
| GraphQL | 这个 API 一共 2 个端点，REST 已经够 |
| 用 Flask / Django | FastAPI 自带 pydantic 校验、async lifespan、OpenAPI 文档，对这种"一个端点 + 模型加载"场景最省事 |
| 启动时 lazy-load 模型（首次请求才加载） | 第一个请求等 10 秒体验差，且容易把模型加载错误掩盖到运行时 |
| 用 BackgroundTasks 异步检索 | 检索是 ~30ms 的同步操作，加 background 反而更慢更复杂 |
| 把模型放进单独 service（gRPC） | 单机部署不需要；想做了再说 |

---

## 8. 改这一层的注意事项

- **改了 request / response schema 记得更新本文档第 2 节**。`/docs`（Swagger）会自动反映 pydantic 模型，但人类要看的是这一份。
- 加 `/answer` 等新接口时，**保持 `/query` 输出格式不变** —— 它是其他系统的契约。
- 如果加了 LLM 路由：把 LLM 配置（base_url / api_key）也走 pydantic-settings + `.env`，保持配置入口统一。
- **永远不要在 response 里返回内部异常 stack**（`detail=str(e)` 已经偏多了，但能接受，因为这是个非生产服务）。

---

下一步：[`operations.md`](./operations.md)（部署、维护、故障排查）
