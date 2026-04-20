# 检索策略（Qdrant Hybrid + RRF）

把 [`embedding.md`](./embedding.md) 输出的 (dense, sparse) 灌进 Qdrant，并在查询时把两路结果用 RRF（Reciprocal Rank Fusion）融合。

代码：`src/terraria_rag/store/qdrant_store.py`，被 `scripts/04_index.py` 写、被 `api/server.py` 读。

---

## 1. 为什么用 Qdrant

- **同一个 collection 同时支持 dense + sparse 向量** —— 这是 hybrid 检索的硬要求，绝大部分老向量库（FAISS、Milvus 早期版本）做不到原生 hybrid。
- **Query API 内置 RRF / DBSF fusion**，不用自己实现融合。
- **Embedded 模式**：传 `path=...` 就用文件持久化，不用 docker、不用 server，对个人项目极友好。同样一份代码改成 `url=...` 就能切到生产模式。
- payload 任意 JSON，把 `title` / `section_path` / `url` 顺手放进去，检索结果不用回查。

---

## 2. Collection schema

```python
client.create_collection(
    collection_name="terraria_zh",
    vectors_config={
        "dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": qm.SparseVectorParams(),
    },
)
```

- 两个**命名向量**：`dense`（1024 维 cosine）+ `sparse`。命名是必须的，因为 Qdrant 用名字来区分查询走哪一路。
- 不显式给 sparse 配 `IDF` 等高级参数 —— BGE-M3 已经给出了 learned weights，sparse 这边就**纯当一份带权倒排存**，让 RRF 在 rank 层面融合就行。
- payload schema（每个点都带）：

```python
{
    "pageid": int,
    "title": str,
    "section_path": str,
    "chunk_index": int,
    "text": str,            # 原文，直接返回给调用方
    "url": str,             # 拼好的页面链接
}
```

> ⚠️ 当前**没建 payload index**。如果以后要做 `filter=Filter(must=[FieldCondition(key="categories", ...)])` 这种分面过滤，要给对应字段建索引（`create_payload_index`），否则全 scan。

---

## 3. Point ID 设计

```python
@staticmethod
def _point_id(pageid: int, chunk_index: int) -> int:
    return (int(pageid) << 12) | (int(chunk_index) & 0xFFF)
```

把 `(pageid, chunk_index)` 打包成一个 64-bit int 当 point id。好处：

- **upsert 自然幂等**：同一个 chunk 再算一遍向量重新 upsert，覆盖原点而不是重复入库。
- **不用维护额外的 id 表**：从 chunk 元数据就能算出 id，反向也能从 id 算回 (pageid, chunk_index)。
- **稳定**：增删别的 chunk 不影响这个点的 id。

约束：`chunk_index < 4096`（12 bit）。terraria.wiki.gg 上最长页面 ~50 chunks，离上限远。如果哪天爬整本《魔戒》全文，再扩 bit。

---

## 4. 写入：`upsert(chunks, embeddings)`

```python
points.append(qm.PointStruct(
    id=pid,
    vector={
        "dense": emb.dense,
        "sparse": qm.SparseVector(indices=..., values=...),
    },
    payload=payload,
))
self.client.upsert(collection_name=..., points=points)
```

- 一次 upsert 一个 batch（`scripts/04_index.py` 默认 32）。Qdrant embedded 模式 IO 是顺序的，batch 大小对吞吐影响有限，主要是减少 Python 调用开销。
- `upsert` 而不是 `insert`：再跑一次脚本不会因为 id 冲突报错。

---

## 5. 查询：`hybrid_search(dense, sparse, top_k)`

核心是 Qdrant Query API 的 **prefetch + fusion** 模式：

```python
self.client.query_points(
    collection_name=...,
    prefetch=[
        qm.Prefetch(query=dense,                 using="dense",  limit=prefetch_k),
        qm.Prefetch(query=qm.SparseVector(...),  using="sparse", limit=prefetch_k),
    ],
    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
    limit=top_k,
    with_payload=True,
)
```

执行步骤：

1. 在 dense 字段上跑 ANN 取 top `prefetch_k`。
2. 在 sparse 字段上跑倒排取 top `prefetch_k`。
3. 把两路结果**按排名**做 RRF 融合，输出 top `top_k`。

### 5.1 `prefetch_k` 的取值

```python
prefetch_k = max(top_k * 4, 32)
```

- 太小：两路结果交集少，RRF 退化成"取并集打分"，融合优势消失。
- 太大：每路都拉太多，浪费。
- `4×` 是 RRF 论文经验值，下限 32 是为了 top_k 很小时（如 1）也有足够候选。

### 5.2 RRF 是什么、为什么用它

Reciprocal Rank Fusion 公式：

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 d 在第 i 路结果中的排名（1-based），`k` 是平滑常数（Qdrant 默认 60）。

为什么 RRF 比"加权求和 dense_score + α·sparse_score"好：

- **不用调 α**。dense cosine 一般在 [−1, 1]，sparse score 量纲完全不同，加权前必须 z-score / min-max 归一化，做错就翻车。
- **对绝对分数不敏感**，只看排名 —— 鲁棒性好。
- **对两路检索器质量差距宽容**：哪怕一路噪声大，只要它能把真正相关的文档排进前几名，RRF 就能加权。
- 最近的 BEIR / MTEB 综述里 RRF 在 hybrid 里基本是 SOTA baseline。

### 5.3 备选：`Fusion.DBSF`

Qdrant 也支持 Distribution-Based Score Fusion（DBSF），它会做 z-score 归一后加权。理论上效果可能略好，但需要更多文档拟合分布；对我们这种几十个 chunk 的小 top_k 场景，RRF 更稳。想试就一行：

```python
query=qm.FusionQuery(fusion=qm.Fusion.DBSF),
```

---

## 6. 返回值：`RetrievedChunk`

```python
@dataclass
class RetrievedChunk:
    score: float          # RRF 融合后的得分（不是 cosine！）
    pageid: int
    title: str
    section_path: str
    text: str
    url: str
```

注意 `score` 的语义：

- 是 **RRF 分数**，量级一般在 0.01–0.05 之间。
- **不可跨查询比较**：同一个 query 的 hits 之间分数可比，不同 query 之间没意义。
- 不要把它当 cosine 用做"相似度阈值过滤"。要做阈值过滤，关掉 fusion 单跑 dense。

---

## 7. URL 拼接

```python
def _page_url(title: str) -> str:
    safe = title.replace(" ", "_")
    return f"{settings.page_url_prefix}/{safe}"
```

`page_url_prefix = "https://terraria.wiki.gg/zh/wiki"`。空格转下划线是 MediaWiki URL 规范。

不去做完整的 percent-encoding 是因为：

- 中文标题主流浏览器都能直接处理（实测 chrome / firefox / safari 都行）。
- 加 `urllib.parse.quote` 反而让人看链接看不懂。

如果要严格点（怕 `%` `&` `?` 出现在 title 里），改一行加个 `quote(safe, safe='/')` 即可。

---

## 8. 性能数字

实测 ~50k chunks 的 collection（embedded 模式，Apple M2 Pro NVMe SSD）：

| 操作 | 耗时 |
|---|---|
| 单次 hybrid 查询（top_k=8, prefetch_k=32） | ~30 ms |
| 加 BGE-M3 query encode（cuda） | ~30+30 = 60 ms |
| 加 BGE-M3 query encode（cpu） | ~30+800 = 830 ms |
| upsert 一批 32 chunks | ~50 ms |
| count 全量 | ~5 ms |

bottleneck 永远是 query encode，**不是 Qdrant 本身**。生产里如果想压低延迟，要么把 BGE-M3 上 GPU、要么前置一个 query embedding cache。

---

## 9. 不采用的方案 & 原因

| 方案 | 为什么没用 |
|---|---|
| 只用 dense（不要 sparse） | 中英术语并存场景 + 数字 / 版本号查询命中率明显下降 |
| 只用 sparse（BM25-only） | 跨语言 / 同义改写召不回来 |
| 加权求和（α·dense + (1-α)·sparse） | 量纲问题 + α 难调 + 对模型分数分布敏感 |
| 加 cross-encoder reranker（默认） | top-8 已经够准；启动 reranker 模型多 ~1G 内存 + 增加 100ms 延迟。**留在依赖里随时可加**（见下） |
| FAISS HNSW + 单独 BM25 | 两个组件分开维护，hybrid 融合得自己写 |
| Qdrant server（docker） | embedded 已经够用，少一个运维负担 |

---

## 10. 加 reranker（如果效果不够）

依赖里已经有 `FlagEmbedding`（同包提供 `FlagReranker`）。改 `api/server.py` 大概这样：

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# in /query handler:
hits = store.hybrid_search(q.dense, q.sparse, top_k=top_k * 5)  # 多召一些
pairs = [[req.query, h.text] for h in hits]
scores = reranker.compute_score(pairs, normalize=True)
hits = [h for _, h in sorted(zip(scores, hits), reverse=True)][:top_k]
```

**什么时候加**：用了一段时间发现 top-1 经常不是最佳答案、且 top-3/5 里有正确答案 —— 这时 reranker 能精排明显有效。

**什么时候不加**：top-1 已经常对、或者你的应用允许 LLM 看 top-8 自己选 —— reranker 是无谓开销。

---

## 11. 改这一层的注意事项

- **改了 schema（dense_dim、加 sparse 字段名）必须 `--rebuild`**。
- Qdrant embedded 模式**同时只能一个进程打开 `data/qdrant/`**：`04_index.py` 还在跑就别开 `05_serve.py`。
- 想在另一台机器查询 —— 不要直接拷 `data/qdrant/`，要么走 server 模式 + snapshot，要么从 `chunks.jsonl` 重跑 `04_index.py`。
- 加 payload filter 之前**先建 payload index**，否则查询会全扫。

---

下一步：[`api.md`](./api.md)（FastAPI 怎么把这一切串起来对外）
