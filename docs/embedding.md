# Embedding 策略（BGE-M3）

把 [`chunking.md`](./chunking.md) 输出的每个 `Chunk.text` 编码成"向量 + 学习的稀疏权重"，准备入库做混合检索。

代码：`src/terraria_rag/embedding/bge.py`，被 `scripts/04_index.py` 和 `api/server.py` 复用。

---

## 1. 为什么选 BGE-M3

中文 RAG 的事实标准之一。它一次 forward 同时输出三种表示：

| 表示 | 维度 | 用途 |
|---|---|---|
| **dense** | 1024 维 float | 语义相似度（cosine） |
| **sparse** | token id → weight 的 dict | 类 BM25 的"学习版"，对术语 / 拼写敏感 |
| **colbert** | (n_tokens, 1024) 多向量 | 细粒度匹配，**我们不用** |

为什么这套适合 Terraria wiki：

- **中文强**。BGE-M3 在 C-MTEB / MIRACL-zh 上排第一梯队。
- **dense + sparse 一次出**，避免再装一个独立的 BM25 实现。BGE 的 sparse 还会学到"哪些 token 重要"，比纯 BM25 强。
- **8K 上下文**（虽然我们配的 1024 已经够 chunk 用）。
- **2.3GB 单模型**，本地能跑，不依赖任何在线服务。

不用 colbert 的原因：

- 多向量存储成本翻 ~5–10 倍。
- Qdrant 需要额外配 `multi_vector` 字段，schema 复杂。
- 我们的 chunks 不长（~512 tokens），dense + sparse 已经够细。

---

## 2. 接口

```python
class BGEM3Embedder:
    DENSE_DIM = 1024

    def encode(self, texts: Iterable[str]) -> list[EmbeddedChunk]: ...
    def encode_query(self, text: str) -> EmbeddedChunk: ...

@dataclass
class EmbeddedChunk:
    dense: list[float]            # 1024 维
    sparse: dict[int, float]      # token_id -> weight，已过滤 weight=0
```

`encode_query` 只是 `encode([text])[0]`，没有"query / document 不对称"的特殊处理——BGE-M3 训练时就是对称的（不像 e5 / bge 系列旧版需要加 `"query: "` / `"passage: "` 前缀）。

---

## 3. 关键配置

| 配置 | 默认 | 含义 |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace repo id 或本地路径（推荐改成本地路径，见 [`operations.md`](./operations.md)） |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps`（Apple Silicon） |
| `EMBEDDING_BATCH_SIZE` | `8` | encode 时的 batch；显存吃紧调小 |
| `EMBEDDING_MAX_LENGTH` | `1024` | 单条文本截断长度（token） |

### 3.1 device 选择

```python
use_fp16 = settings.embedding_device == "cuda"
self.model = BGEM3FlagModel(..., use_fp16=use_fp16, devices=settings.embedding_device)
```

- **CUDA**：必开 `fp16`，~2× 速度，几乎无质量损失。
- **MPS**（Apple Silicon）：fp16 在 PyTorch MPS 后端上**很多算子还不稳**，所以关掉。实测 M2 Pro ~9 chunks/s，5 万 chunks 大概 1.5h。
- **CPU**：纯 CPU 跑得动但慢，~1–3 chunks/s。建议先 `--limit 50` 试跑别一开始上 8000 页。

> 关于 device 字符串：`FlagEmbedding` 接受 `"cpu"` / `"cuda"` / `"cuda:0"` / `"mps"`。多卡传 `["cuda:0", "cuda:1"]`，但 `BGEM3Embedder` 当前没暴露多卡接口（够用，懒得加）。

### 3.2 batch_size

显存敏感参数。粗略估算（fp16）：

| device | 推荐 `EMBEDDING_BATCH_SIZE` | 备注 |
|---|---|---|
| RTX 3060 12GB | 32 | 安全 |
| RTX 4090 24GB | 64–128 | 可激进 |
| M2 Pro 16GB | 8–16 | MPS 内存共享，不能太贪心 |
| 纯 CPU | 4–8 | 大也无益（CPU 已经是 bottleneck） |

调大的收益是**减少 forward 调用次数**，主要省 Python 调用 / GPU launch 开销，对总耗时影响通常 ~10–30%。

### 3.3 max_length

`1024` 是经过权衡的：

- BGE-M3 训练支持 8192，但实测**长文本质量提升边际**且显存 / 时间开销线性增长。
- 我们的 chunk 设计目标就是 512 tokens 左右（见 [`chunking.md`](./chunking.md)），1024 留 2× 安全边际。
- query 短得多（用户问句一般 < 50 tokens），同样 max_length=1024 完全够。

**不要调小到 < 512**——会把长 chunk 的尾巴截掉，造成"明明库里有，却搜不到"的 silent failure。

---

## 4. dense vs sparse 都各扮什么角色

### dense（语义向量）

擅长：

- "如何获得飞行员风镜" 找到 "在沙漠中由秃鹰掉落"（query 和命中文本零字面重合）
- 同义改写、近义概念

弱点：

- 拼错就直接拉低相似度
- 罕见专名（数字 ID、版本号、模板名）的区分度不如稀疏

### sparse（学习的 BM25）

擅长：

- 命中专名 / 数字（"1.4.4"、"item infobox"）
- 罕见词的高权重
- 用户和文档使用相同术语时的强信号

弱点：

- 完全不懂语义，"克苏鲁之眼" 找不到 "Eye of Cthulhu"（我们刻意保留的英文术语）

### 为什么必须二者皆要

terraria.wiki.gg 上**中文译名 + 英文原名**并存。dense 处理跨语言、跨表达；sparse 处理专名、数字、版本号。两者交集 + RRF fusion 见 [`retrieval.md`](./retrieval.md)。

---

## 5. sparse 的存储格式

BGE-M3 输出的 `lexical_weights` 是 `{token_id_str: weight}`：

```python
{"6": 0.34, "10": 0.21, "1234": 0.05, ...}  # 都是 str→float
```

我们做两件事：

```python
sparse = {int(k): float(v) for k, v in lw.items() if v > 0}
```

1. **key 转 int**：Qdrant 的 `SparseVector` 要 `indices: List[int]`。
2. **过滤 weight=0**：BGE 偶尔会输出 0 权重的 token，存了浪费空间。

每个 chunk 的 sparse 一般有 10–80 个非零 token（取决于文本长度和重复度）。

---

## 6. 一些数字（性能 & 质量参考）

实测 BGE-M3 在 terraria.wiki.gg/zh ~50k chunks 上的开销：

| 阶段 | 耗时（M2 Pro mps）| 耗时（RTX 4060 cuda+fp16）|
|---|---|---|
| 加载模型 | ~12s | ~8s |
| encode 50k chunks | ~90 min | ~10 min |
| encode 单条 query | ~80 ms | ~30 ms |

dense 向量大小：50k × 1024 × 4 bytes = **~200 MB**
sparse 平均 ~30 个非零项：50k × 30 × 8 bytes ≈ **~12 MB**
最终 Qdrant 落盘大小（含 payload + 索引）：**~400 MB** 量级

---

## 7. 不采用的方案

| 方案 | 为什么没用 |
|---|---|
| `text-embedding-3-small` (OpenAI) | 收费 + 中文不如 BGE-M3 + 不能离线 |
| `BAAI/bge-large-zh` | 老一代单 dense，没有 sparse + 没有跨语言能力 |
| `m3e-base` | 中文不错但社区已经基本被 BGE-M3 取代 |
| `colbert-v2` | 多向量存储成本高，本场景不需要这么细 |
| 自己训 dual encoder | 没有标注数据，成本极高 |
| 用 `sentence-transformers` 直接调 | BGE-M3 推荐用 `FlagEmbedding` 包，sparse 接口更原生 |

---

## 8. 改这一层的注意事项

- **换模型必须 `04_index.py --rebuild`**。dense 维度变了 / sparse 词表变了，旧 collection 直接废。
- 改 `EMBEDDING_MODEL` 路径时记得**同时改 `DENSE_DIM`**。BGE-M3 是 1024，bge-large 是 1024，bge-base 是 768，e5-large 是 1024……不要想当然。
- 如果加 reranker（`bge-reranker-v2-m3`），它**不影响**这一层。reranker 在 retrieval 之后做，见 [`retrieval.md`](./retrieval.md)。
- 模型本地化下载强烈推荐放进 `./models/bge-m3`，避免每次启动都触网检查 HF。具体见 [`operations.md`](./operations.md)。

---

下一步：[`retrieval.md`](./retrieval.md)（怎么把这两种向量塞进 Qdrant 并 hybrid 查询）
