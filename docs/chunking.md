# 切块（Chunking）策略

把 [`cleaning.md`](./cleaning.md) 输出的 `Section` 列表切成可以喂给 BGE-M3 的 `Chunk`，每个 chunk 自带 page / section 元数据用来回引。

代码：`src/terraria_rag/chunking/splitter.py`，被 `scripts/03_clean_chunk.py` 调用。

---

## 1. 设计目标

| 目标 | 怎么实现 |
|---|---|
| 一个 chunk = 一个语义单元 | **优先以 wiki section 为单位**，而不是机械切窗口 |
| chunk 不能超过模型 max_length | 长 section 二次切（段落优先 → 硬截断 fallback） |
| 即使切散了也要保留上下文 | chunk 文本前缀加 section breadcrumb，相邻 chunk 加 overlap |
| 检索结果能回引到原文位置 | 每个 chunk 带 `pageid`、`title`、`section_path`、`chunk_index` |
| 不依赖 tokenizer | 字符数估算（中文 1 token ≈ 1.6 字符） |

---

## 2. 数据结构

```python
@dataclass
class Chunk:
    pageid: int
    title: str
    section_path: str   # "泰拉之靴 > 制作 > 配方"
    text: str           # 真正喂给 embedding 的文本
    chunk_index: int    # 同一页内第几个 chunk
```

`(pageid, chunk_index)` 是 chunk 的天然主键。`store/qdrant_store.py` 把这两个数压成一个 64-bit point id（`pageid << 12 | chunk_index`），见 [`retrieval.md`](./retrieval.md)。

`section_path` 是从根开始的 breadcrumb，**直接拼进 chunk 文本里作为前缀**：

```
# 泰拉之靴 > 制作 > 配方

在工作台合成。
- 火箭靴 + ...
```

为什么前缀进去：BGE-M3 是个 ~568M 参数的小模型，它**没有 attention 之外的 metadata 通道**。把 breadcrumb 写进文本里，是让模型在算 embedding 时能"知道这段是什么主题下的"。同样的句子在 `泰拉之靴 > 制作` 下和 `飞行员风镜 > 笔记` 下应该有不同的语义表示。

---

## 3. 算法

```
sections (from cleaning)
    │
    ▼ _build_section_paths(sections)
[(section, "PageTitle > ... > heading"), ...]
    │
    ▼ for each section:
    │     budget = max_chars - len(prefix)
    │     pieces = _split_long(section.body, budget, overlap_chars)
    │     for each piece:
    │         emit Chunk(text=prefix + piece, ...)
    ▼
list[Chunk]
```

### 3.1 Breadcrumb 重建（`_build_section_paths`）

输入 sections 是扁平列表（cleaning 里故意打平了），但每个 section 自己知道 `level`，所以用栈重建层级：

```python
stack: list[(level, heading)] = []
for s in sections:
    while stack and stack[-1].level >= s.level:
        stack.pop()
    stack.append((s.level, s.heading))
    paths.append(" > ".join(h for _, h in stack))
```

例：

| Section | stack 状态 | path |
|---|---|---|
| `(1, "泰拉之靴")` | `[(1, "泰拉之靴")]` | `泰拉之靴` |
| `(2, "制作")` | `[(1, "泰拉之靴"), (2, "制作")]` | `泰拉之靴 > 制作` |
| `(3, "配方")` | `[(1, "泰拉之靴"), (2, "制作"), (3, "配方")]` | `泰拉之靴 > 制作 > 配方` |
| `(2, "笔记")` | `[(1, "泰拉之靴"), (2, "笔记")]` | `泰拉之靴 > 笔记` |

### 3.2 长 section 二次切（`_split_long`）

预算超了就切。**三档 fallback**：

1. **段落级**（`\n\n` 分隔）：尽量把整段塞进一个 chunk。
2. **段落自己就超长**：硬截断 `[i : i+max_chars]`，相邻片段共享 `overlap_chars` 个字符。
3. **加 overlap**：除了第一个 chunk，每个 chunk 在开头粘上前一个 chunk 的尾巴 `overlap_chars` 字符。

为什么不做"句子级"切：

- 中文句号 `。` 不一定在每段都有（列表、表格、代码块都没有）。
- BGE-M3 max_length 是 1024 tokens，配额已经很宽，按段落切几乎不会触底。
- 句子边界检测引入 `jieba` / `pkuseg` 这种额外依赖，性价比不高。

### 3.3 字符 vs token

用 `_CHARS_PER_TOKEN_CN = 1.6` 把 token 预算换成字符预算：

```python
max_chars = int(max_tokens * 1.6)        # 默认 512 * 1.6 = 819
overlap_chars = int(overlap_tokens * 1.6) # 默认 64 * 1.6 = 102
```

为什么不用 tokenizer：

- 加载 tokenizer = 加载 BGE-M3 spm 模型，**~400ms 启动开销 × 切块脚本只跑一次**，不太值。
- 中文 token 数估计很稳：常用字几乎都是 1 token，标点和英文略宽，1.6 是经验中位数。
- 真的超了 BGE-M3 内部 `max_length` 还会兜底截断，不会崩。

代价：英文 / 代码块的页面（terraria.wiki.gg 上少数）字符 → token 比例更小，可能造成 chunk 偏小。可接受。

### 3.4 Prefix budget 保护

```python
prefix = f"# {path}\n\n"
budget = max_chars - len(prefix)
if budget <= 200:
    prefix = ""
    budget = max_chars
```

万一某个 section path 异常长（嵌套很深 + heading 很长），保住 body 的最低预算 200 字符。极端情况就放弃 prefix——总比一个空 chunk 强。

---

## 4. 配置

`src/terraria_rag/config.py` 里：

| 配置 | 默认 | 含义 |
|---|---|---|
| `CHUNK_MAX_TOKENS` | `512` | 每个 chunk 的 token 预算（含 prefix） |
| `CHUNK_OVERLAP_TOKENS` | `64` | 相邻 chunk 重叠的 token 数 |

调参建议：

- **检索召回率低**：`CHUNK_MAX_TOKENS` 调到 `256`，把 chunks 切得更细，单 chunk 主题更聚焦。
- **答案被切碎**（一个事实横跨两个 chunk 都答不全）：`CHUNK_OVERLAP_TOKENS` 调到 `128`。
- **存储 / embedding 时间太长**：调大 `CHUNK_MAX_TOKENS` 到 `768`，chunk 数会显著减少。

不建议超过 `1024`：BGE-M3 的训练配置就是 8192，但 [官方说明](https://huggingface.co/BAAI/bge-m3) 指出**长上下文质量和 1024 区别不大、显存 / 时间开销显著**。

---

## 5. 输出体量预估

terraria.wiki.gg/zh 实测（仅供参考）：

| 数量 | 数量级 |
|---|---|
| pages（爬下来的） | ~5000–8000 |
| sections（清洗后） | 平均 5–8 / page → ~30k–60k |
| chunks（切完） | 平均 1.2× sections → ~40k–80k |

输出 `data/cleaned/chunks.jsonl` 体量：~50–100 MB。

---

## 6. 一个完整的 chunk 长这样

```json
{
  "pageid": 1234,
  "title": "泰拉之靴",
  "section_path": "泰拉之靴 > 制作",
  "chunk_index": 1,
  "text": "# 泰拉之靴 > 制作\n\n泰拉之靴可以通过下述配方在工作台合成。\n\n[item infobox]\n- 名称: 泰拉之靴\n- 类型: 饰品\n- 防御力: 0\n..."
}
```

注意：

- `text` 是**最终喂给 embedding 的文本**，已经包含 prefix 和清洗后的内容。
- `pageid` + `chunk_index` 唯一标识。
- `title` / `section_path` **冗余存了一份**，用来在 Qdrant payload 里直接展示，不用回查 jsonl。

---

## 7. 不采用的方案

| 方案 | 为什么没用 |
|---|---|
| 固定窗口滑动切（如每 512 tokens 切一刀） | 会把 infobox 切两半，把"配方"标题和配方内容切散，召回质量差 |
| 用 LangChain 的 `RecursiveCharacterTextSplitter` | 多一个依赖、行为不透明、对中文标点支持一般。我们这套足够简单 |
| LLM 辅助切块（让 LLM 决定边界） | 又慢又贵；wiki 已经有人工 section 划分，没必要再让 LLM 重做 |
| 不加 section breadcrumb | 同样的句子在不同主题下 embedding 区分不开，相关性下降 |
| 加 token-level overlap（精确按 token 切） | 引入 tokenizer 依赖，复杂度上升；字符级 overlap 已经够用 |

---

## 8. 改这一层的注意事项

- 改 `CHUNK_*` 后**必须重跑** `03_clean_chunk.py` + `04_index.py --rebuild`。
- `_CHARS_PER_TOKEN_CN` 是中文站经验值。换 en 站建议改成 `4.0`（1 token ≈ 4 chars，英文 BPE 经验值）。
- `chunk_index` 字段被 `qdrant_store.py` 编码进 point id：**单页 chunk 数必须 < 4096**（`& 0xFFF` 限制）。terraria.wiki.gg 上最长的页面也就 ~50 chunks，离上限远得很。

---

下一步：[`embedding.md`](./embedding.md)（怎么把 chunks 编码成向量）
