# 简历 & 面试备战（Terraria RAG）

面向 **LLM / RAG / AI 应用工程师** 岗位的精简版。

> 一句话定位：**端到端、可离线运行的中文垂直领域 RAG 系统**。从爬取 → 清洗 → 切块 → 混合向量化 → 检索 → API，全部自研，技术选型每一步都讲得出 trade-off。
>
> 这正是面试官想听的——不是"调了 LangChain"，而是"我懂每一层在做什么，为什么这么选"。

---

## Part 1 · 简历怎么写

### 1.1 项目标题（任选一个）

- **Terraria Wiki 中文垂直领域 RAG 系统**
- **基于 BGE-M3 + Qdrant 的中文 Wiki 知识库 RAG（端到端自研）**
- **离线可运行的中文 RAG 系统：MediaWiki 爬取 → 混合检索 → FastAPI**

### 1.2 简洁版 bullets（4 行，推荐）

> 用 STAR-lite：技术栈 + 量化结果 + 工程亮点。每行一个亮点，避免一行塞两件事。

```
Terraria Wiki 中文 RAG 系统 | Python · BGE-M3 · Qdrant · FastAPI    [GitHub]
- 端到端自研中文垂直领域 RAG（MediaWiki 爬取 → wikitext 清洗 → section 切块 → 混合检索 → API），全程本地可离线运行，零云端依赖。
- 通过 MediaWiki API 批量化（一次请求 50 titles）+ 线程池并发 + 全局令牌桶限速，将 8000+ 页面爬取从 ~2.2h 优化到 ~10min（提速 ~150×），自带断点续传与原子写。
- 检索层用 BGE-M3 同时产出 dense (1024d) + sparse 表示，配合 Qdrant Query API 的 RRF fusion 做混合检索，解决中英术语并存场景下纯稠密召回不全的问题。
- 切块策略按 wiki section 切分 + 注入 breadcrumb 前缀（"页 > H2 > H3"），让模型在 embedding 阶段获得结构化上下文，5 万级 chunks 单次 hybrid 查询 ~30 ms。
```

### 1.3 极简版（2 行，给一页简历用）

```
Terraria Wiki 中文 RAG | Python · BGE-M3 · Qdrant · FastAPI
端到端自研中文 RAG：MediaWiki 批量爬取（提速 ~150×）→ wikitext 结构化清洗 → BGE-M3 dense+sparse 混合向量化 → Qdrant 本地化部署（RRF 融合）→ FastAPI。
5 万 chunks 规模下 hybrid 查询 ~30ms，全离线可跑。
```

### 1.4 写简历时的几条原则

1. **写技术名词要够具体**：写 "Qdrant + RRF" 比写 "向量数据库 + 融合算法" 强 10 倍——前者面试官可以追问，你能接住。
2. **量化要诚实**：8000 页、~150× 提速、30ms 查询、~50k chunks 都是真实数据；别写"提升 90%"这种没基线的数字。
3. **"端到端"是关键词**：现在做"调 LangChain 拼一个 demo"的人太多，"全栈自研 + 讲得清 trade-off"是稀缺能力。
4. **不要写 LLM 答案生成**：项目刻意只做检索层不接 LLM（见 [`api.md`](./api.md) 第 1 节），简历里也别瞎吹。被问"那你接的什么 LLM"会很尴尬。
5. **GitHub 链接放上**。RAG 岗位面试官 **80%** 会去看代码。代码本身就是简历的延伸（你有完整的 `docs/`，这点很加分）。

### 1.5 STAR 详细版（如果有人让你"讲讲这个项目"）

| 段 | 30 秒讲稿 |
|---|---|
| **S**ituation | 想做一个能离线运行的中文 Terraria 知识助手，市面上没有——目标知识库就是 terraria.wiki.gg/zh，~8000 页 wikitext。 |
| **T**ask | 端到端搭一套 RAG：把 wiki 爬下来、清洗成可检索的形式、向量化、提供查询接口；要求**单机可跑、零云端依赖、可断点续传**。 |
| **A**ction | 5 步流水线，每步落盘解耦：(1) `01_enumerate` 用 `list=allpages` 列页；(2) `02_crawl` 批量+并发拉 wikitext，提速 ~150×；(3) `03_clean_chunk` 用 `mwparserfromhell` 解析 + 按 section 切块，模板分三类策略处理；(4) `04_index` 用 BGE-M3 同时算 dense + sparse，灌进 Qdrant embedded；(5) `05_serve` FastAPI 暴露 `/query`，内部用 Qdrant 的 RRF fusion 做混合检索。 |
| **R**esult | 8000 页爬取 ~10 分钟，~50k chunks 单次查询 ~30ms，~3GB RAM 单机部署。每个模块都写了策略文档（`docs/`），讲清"为什么这么选"。 |

---

## Part 2 · 高频面试题（20 题，按 RAG 工程师面试热度排序）

> 顺序 = 重要性。前 10 题是"必答"，后 10 题是"加分项"。每题给"考官在考什么"+ "怎么答（提纲）"。

---

### Q1. 为什么用混合检索（dense + sparse）？只用 dense 不行吗？

**考点**：你是否真的理解两种召回的边界。

**怎么答**：

- dense 擅长**语义相似 / 同义改写 / 跨语言**（"如何获得风镜" → "秃鹰掉落"）。
- dense 弱在**专名 / 数字 / 罕见术语**（"克苏鲁之眼" 找不到 "Eye of Cthulhu"，因为我项目里刻意保留了英文术语，见 `cleaning.md`）。
- sparse（BGE-M3 学习的 BM25）正好补这个洞：对字面命中和高权重 token 敏感。
- 真实场景里 query 风格五花八门，**只用一种召回都会有 silent failure**——能召回但你不知道漏了什么。
- Hybrid 不是"二选一"而是"两路并行召回 + 融合"，召回率上限明显更高。

---

### Q2. 为什么用 RRF 而不是加权求和？

**考点**：分数融合的常识。这是 RAG 工程师必知。

**怎么答**：

- 加权求和 `α·dense + (1-α)·sparse` 的问题：
  1. **量纲不同**：cosine 在 [-1,1]，sparse score 量级完全不同，必须先归一化。
  2. **α 难调**：不同 query / 不同 corpus 最优 α 不一样。
  3. 对**绝对分数分布敏感**，模型升级后可能整套阈值要重调。
- RRF 公式 `score(d) = Σ 1/(k + rank_i(d))`，**只看排名不看分数**。
  1. 不用归一化、不用调 α；
  2. 哪怕一路召回器质量差，只要它能把对的文档排进前几位，RRF 就能加权；
  3. BEIR / MTEB 综述里 RRF 是 hybrid 的 SOTA baseline。
- Qdrant 的 Query API 原生支持 `Fusion.RRF`，一行代码搞定。

---

### Q3. BGE-M3 对比 OpenAI text-embedding-3 / m3e / bge-large 的优势？

**考点**：embedding 选型理由。

**怎么答**：

- **一次 forward 同时出 dense + sparse + colbert 三种表示**——做混合检索不用再装一个 BM25 库。
- **中文和跨语言强**：C-MTEB / MIRACL 上中文第一梯队，跨语言 zero-shot 能力对"中英术语并存"特别合适。
- **8K 上下文**（虽然我项目用 1024 已经够）。
- **本地可跑、~2.3GB**，无 OpenAI 那种"必须联网 + 收费 + 数据出境合规"问题。
- 跟旧版 bge-large-zh 比：最大区别是有了 sparse + 跨语言；跟 m3e 比：BGE-M3 是社区现在的事实标准。
- 不用 colbert（多向量）的原因：存储成本翻 5–10 倍，本场景 chunk 不长（~512 tokens），dense + sparse 已经够细。

---

### Q4. 你的切块策略是什么？为什么不用固定窗口滑动？

**考点**：chunking 是 RAG 最容易翻车的一环，看你有没有想清楚。

**怎么答**：

- 优先**按 wiki 自带的 section 切**，每个 section 是一个语义自然单元；长 section 才二次切（段落 → 硬截断 fallback + overlap）。
- **切完往 chunk 文本前缀里塞 breadcrumb**（`# 页 > H2 > H3\n\n...`），让 BGE-M3 在算 embedding 时能"知道这段在什么主题下"，同样的句子在不同章节下应有不同表示。
- 不用固定窗口的原因：会把 infobox 切两半、把"配方"标题和配方内容切散，**召回时能命中关键词但答非所问**。
- chunk 大小用**字符估算**（中文 1 token ≈ 1.6 字符），不引入 tokenizer 依赖；BGE-M3 内部 `max_length` 兜底真超了会截断。

---

### Q5. 单页太长怎么办？相邻 chunk 要不要 overlap？

**考点**：长上下文处理 + 切边界一致性。

**怎么答**：

- 三档 fallback：段落级（`\n\n`）→ 硬截断 → 加 overlap。
- 默认 `overlap = 64 tokens ≈ 102 字符`：相邻 chunk 共享尾部，避免**事实横跨边界时召回不全**。
- 不做"句子级"切的原因：中文标点不规整（列表、表格、代码块没句号），引入 jieba/pkuseg 性价比不高。
- overlap 不是越大越好——**会膨胀向量数 + 重复入库降低 RRF 区分度**。64–128 tokens 是经验甜区。

---

### Q6. Qdrant 用的是什么模式？为什么不用 Pinecone / Milvus？

**考点**：向量库选型 + 部署形态认识。

**怎么答**：

- 我用的是 Qdrant **embedded 模式**（`QdrantClient(path=...)`），数据写到本地 `./data/qdrant/`，**进程内嵌、无 docker、无网络**。生产化只要把 `path=` 改成 `url=` 就切到 server 模式（项目里有现成 `docker-compose.yml`）。
- 选 Qdrant 不选别的：
  - **同一 collection 原生支持 dense + sparse**——这是 hybrid 检索的硬要求，FAISS、老版 Milvus 都做不到。
  - **Query API 内置 RRF / DBSF**，融合不用自己写。
  - payload 任意 JSON，把 title / section / url 顺手存了，检索结果零回查。
- 不选 Pinecone：付费 + 必须联网 + 个人项目过度。
- 不选 FAISS：纯向量库，没 sparse / payload / hybrid，要自己拼一堆胶水。

---

### Q7. embedded 模式有什么坑？怎么扩到生产？

**考点**：你能不能预见到部署陷阱。

**怎么答**：

- **同一时间只能一个进程打开 `./data/qdrant/`**：`04_index.py` 在跑就别开 `05_serve.py`，否则报锁。
- 生产化路径：
  1. `docker compose up -d qdrant` 起 server。
  2. `QdrantClient(url="http://localhost:6333")` 替换。
  3. 多 worker 也才不会互相锁；想跨机就直接暴露 6333。
- 多 worker 时 BGE-M3 模型 ~2.3GB × N 进程会爆 RAM——可以独立部署一个 embedding microservice（`POST /embed`）让所有 worker 共享。

---

### Q8. 为什么 `/query` 不直接接 LLM 出答案？

**考点**：API 边界设计意识。

**怎么答**：

- LLM 选择是**用户的事**：OpenAI / DeepSeek / Claude / Ollama / 通义……每家 SDK、计费、prompt 风格都不一样。硬绑一家排斥所有其他用户。
- 检索质量和生成质量**正交**，解耦后能独立迭代评测。
- 如果以后要接，**新加 `/answer` 路由调 `/query` 拿 chunks 再喂 LLM**，永远保持 `/query` 是纯检索语义。
- 这也是工业界常见做法：检索做成基础设施，generation 做成业务层。

---

### Q9. RAG 系统怎么评测？你这个项目怎么知道效果好？

**考点**：会不会做评估，这是 RAG 工程师的"手艺活"。

**怎么答**：

- **三个层面分开评**：
  1. **Retrieval**：Recall@k、MRR、nDCG。需要 query–相关 chunk 标注集。
  2. **Generation**：忠实度（faithfulness/groundedness）、答案相关性、有无幻觉。
  3. **End-to-end**：人工打分 1–5。
- 工具：**RAGAS**、**TruLens**、**LangSmith** 都可以。RAGAS 提供 faithfulness / answer_relevancy / context_precision / context_recall 四个核心指标，对 RAG 最对症。
- 我这个项目当前**没做正式评测**（坦诚交代），但在 docs 里写了观察方法（`cleaning.md` 第 6 节给了诊断脚本）；下一步会用 LLM-as-judge 造一批 query–truth pair 自动跑 Recall@5。
- **追问反例**：我能讲出"如果只用 dense 会漏哪些 query 类型（专名 / 数字 / 版本号），所以才上 sparse"——这是**定性评估**，比拍脑袋强。

---

### Q10. 怎么处理多轮对话 / query rewrite？

**考点**：进阶 RAG 模式知识。

**怎么答**：

- 我项目当前是单轮检索，没做。但要做的话标准做法：
  1. **Query Rewrite**：把"它有什么用？"这种带指代的 query 用 LLM 改写成"飞行员风镜有什么用？"再去检索（HyDE 也是这个家族）。
  2. **多轮历史压缩**：把对话历史用 LLM 总结成"上下文摘要 + 当前问题"。
  3. **Multi-Query**：用 LLM 把一个问题拆成多个子问题分别检索再合并（适合复杂 query）。
- 这几招都依赖 LLM，所以要么放在 `/answer` 路由里实现，要么在调用方做。
- **代价**：每个 query 多 1–2 次 LLM 调用，延迟 + 成本。

---

### Q11. 你怎么爬 wiki 的？为什么不用 Scrapy？

**考点**：你做的是"程序员的爬虫"还是"脚本小子的爬虫"。

**怎么答**：

- 用 **MediaWiki API**（`/api.php`），不爬渲染后的 HTML。原因：
  - HTML 不稳定（皮肤 / JS 变更），wikitext 是数据原文。
  - 章节切分按 `==` 比解析 `<h2>` 干净。
  - infobox 模板是显式语法，清洗时可以选择性保留。
- 不用 Scrapy 的原因：MediaWiki API 本来就给程序用，不需要中间件 / 选择器那套，小项目用 `httpx + tenacity` 200 行代码搞定。
- 礼貌爬取细节（**面试官特别喜欢追问这块**）：
  - User-Agent 带联系邮箱，被运维盯上时能联系到你。
  - 全局令牌桶 `RateLimiter`（线程安全，进程级）+ 重试用指数退避（2→4→8→16→32s 上限 60s）。
  - 业务错误（`{"missing":true}`）不重试，只重试网络层错误。

---

### Q12. 8000 页爬取从 2 小时优化到 10 分钟，怎么做的？

**考点**：性能优化的系统思维。

**怎么答**（按收益排序）：

1. **批量化最大头**：MediaWiki API 的 `titles` 参数支持 `|` 分隔，匿名上限 50。从 1 title/req → 50 titles/req，**直接 ~50× 加速**。
2. **并发**：`ThreadPoolExecutor(max_workers=4)` 让等服务端响应的时间能重叠。`RateLimiter` 是进程级线程安全的，**全局 RPS 不超**，并发只是把"等"的时间叠加利用。
3. **限速合理化**：从默认 1 RPS 提到 3 RPS（小 wiki 可承受范围）。
4. 写盘改 `tmp + replace` 原子写，去掉 `OPT_INDENT_2`（节省 ~30% 体积）。
- 综合 ~150×。**关键是讲清楚瓶颈分析**：本来卡在"每个请求一个 title × 1 req/s 的限速"，批量解决"每个请求 1 title"，并发 + 提速解决"1 req/s"。

---

### Q13. 限速器是怎么实现的？

**考点**：会不会写线程安全代码。

**怎么答**：

```python
class RateLimiter:
    def __init__(self, rps): self.min_gap = 1.0/rps; self._next_allowed = 0; self._lock = threading.Lock()
    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.min_gap
        if sleep_for > 0: time.sleep(sleep_for)
```

- 关键：**锁内只更新 `_next_allowed`，锁外才 sleep**。否则 sleep 时持锁会导致其他线程全卡死。
- "下次允许时间" 模式比 "记录 last_request_time" 模式好——并发线程预约的是不同的未来时间点，自然排队。

---

### Q14. 如果给你 1000 万页 wiki 怎么办？

**考点**：scale-up 思维。

**怎么答**：

- **爬取**：分布式队列（Celery / RQ / arq），多机器拆 namespace 或 title hash 段。Redis / SQLite 当 visited set。
- **清洗**：embarrassingly parallel，按文件分片用 multiprocessing。
- **embedding**：上 GPU 集群，换 vLLM 或 Triton 部署 BGE-M3，吞吐能到 ~1000 chunks/s/卡。
- **存储**：embedded Qdrant 顶不住，必须切 server 模式 + sharding。10M pages × 平均 5 chunks = 50M chunks × ~1.5KB = ~75GB，单机 SSD 还能扛，但内存索引需要 ~50GB+ 加载，建议用 HNSW 的 on-disk 模式（`on_disk=True`）。
- **索引**：增量索引——爬到一批就 chunk + embed + upsert，不要等全爬完。

---

### Q15. 你怎么处理重定向页？

**考点**：MediaWiki 细节认知。

**怎么答**：

- 枚举时 `apfilterredir=nonredirects` 跳过重定向页本身（它们是 stub）。
- 拉 wikitext 时带 `redirects=1` 让服务端自动跳到目标页。
- 我自己实现了 `title_map: 原始title → normalized → redirected → 最终title`，把响应里 `normalized` 和 `redirects` 数组解析回原始请求 title，**调用方完全不用关心重定向存在**。
- 这一点很多 MediaWiki 客户端做得不干净，会让上层代码看到"我请求 A 但回来的 title 是 B"。

---

### Q16. wikitext 里的模板（`{{...}}`）怎么处理？

**考点**：清洗策略的细致度。

**怎么答**：模板分**三类**处理（细节见 `docs/cleaning.md`）：

1. **直接丢**：`{{ref}}`、`{{nav}}`、`{{stub}}` 这些导航 / 引用 / 维护标签，无语义价值。parser functions（`{{#if:}}`）也丢，客户端没法求值。
2. **展开成 key-value**：infobox / item 这种结构化数据，转成 `[item infobox]\n- 名称: X\n- 攻击力: 20\n` 多行块。**对装备数据查询这是黄金信号**。
3. **取首参数（inline templates）**：`{{tr|Master Mode}}` → `Master Mode`。这点对 terraria.wiki.gg 中文站**至关重要**，否则到处是断句的"在中可获得"。
- 未知模板 fallback 到取首参数，最后用 regex 兜底删残留 `{{...}}`。

---

### Q17. 怎么做增量更新？wiki 改了你怎么知道？

**考点**：生产化思维。

**怎么答**：

- 当前方案是粗粒度增量：`02_crawl.py` 按 `pageid.json` 是否存在决定要不要拉，所以**新增页面会被自动拉**，但**已存在页面的更新不会被检测到**。
- 标准做法：
  1. 用 `prop=info` 拿每个页面的 `lastrevid`，和本地缓存的 `revid` 比对。
  2. 不一样就重拉、重新清洗、重新 chunk、重新 embed、`upsert`（point id 设计成 `(pageid, chunk_index)` 打包，**upsert 自然覆盖旧向量**）。
  3. 旧 chunk 删除：拉之前先记下旧 chunk_index 集合，新一批 upsert 完后 delete 差集。
- 更激进：用 MediaWiki 的 RecentChanges API 增量拿"最近 N 天变更的页面"。

---

### Q18. 流水线 5 步分开的好处和代价？

**考点**：架构设计意识。

**怎么答**：

- **好处**：
  - 每步落盘解耦，相邻步骤通过文件交接，**任意一步失败重跑只跑那一步**。
  - 调试容易：看 jsonl 文件就知道每步输出对不对。
  - 不同步骤可以**独立优化资源**——爬取吃网络，embedding 吃 GPU，互不干扰。
- **代价**：
  - 多次磁盘 IO（jsonl 来回写读）。但我们体量（~50k chunks，~100MB）IO 完全不是瓶颈。
  - 概念上多了"中间产物"要管理。但 `data/raw|cleaned|qdrant` 三层分得很清晰。
- **替代方案**："一个脚本 streaming 跑完"——快但不能断点续传，调试地狱。

---

### Q19. RAG 的常见失败模式有哪些？怎么排查？

**考点**：实战经验。这题答得好直接拉满分。

**怎么答**：

- **召回失败**（要的 chunk 没进 top-k）：
  - 看 query 类型——是专名/数字（sparse 该擅长但没命中）？还是同义改写（dense 该懂但没懂）？分别诊断。
  - 检查 chunk 文本是否真的包含答案（清洗可能把它误删了）。我项目里 `cleaning.md` 第 6 节给了诊断脚本。
  - 调 chunk 大小、调 prefetch_k、加 reranker。
- **召回对了但生成错**（hallucination）：
  - prompt 模板没强调"只用资料回答"。
  - LLM 太弱（小模型对中文长上下文不行）。
  - chunk 信息密度低，LLM 注意力被噪声分散——需要更精细的清洗或 reranker。
- **Lost in the middle**：长 context 中间内容被忽略。解决：top_k 控制在 5–8，重要的放最前 / 最后；或者多级检索 + 摘要。
- **术语漂移**：query 用"克苏鲁之眼"，文档用 "Eye of Cthulhu"。要么靠 sparse、要么前置加 query expansion 字典。

---

### Q20. 如果让你给这个项目加 reranker，你会怎么做？什么时候不该加？

**考点**：知道工具的适用场景，不是无脑堆。

**怎么答**：

- **怎么加**（10 行代码改动）：

  ```python
  from FlagEmbedding import FlagReranker
  reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

  hits = store.hybrid_search(q.dense, q.sparse, top_k=top_k * 5)  # 多召一些
  pairs = [[req.query, h.text] for h in hits]
  scores = reranker.compute_score(pairs, normalize=True)
  hits = [h for _, h in sorted(zip(scores, hits), reverse=True)][:top_k]
  ```

- **加的时机**：观察到 top-1 经常不是最佳答案、且正确答案在 top-3/5 里——精排能立竿见影。
- **不加的时机**：
  - top-1 已经常对——多余开销。
  - 应用允许 LLM 看 top-8 自己挑——LLM 本质上就是 reranker，重复劳动。
  - 延迟敏感场景——reranker 推理 ~100–300ms，加上去就破了 SLO。
- 我项目默认不加是因为：依赖里已经预留好（`FlagEmbedding` 自带 `FlagReranker`），但**先做评测再决定加不加**，比一上来就加更工程师。

---

## Part 3 · 还可能被深挖的 5 个犄角旮旯（加分）

| 问题 | 一句话答案（详情翻 docs/） |
|---|---|
| 为什么用 jsonl 不用 parquet/sqlite？ | 流式可读、可 `head/wc`、断行追加，体量在舒适区，列式查询用不上。见 `architecture.md` §5。 |
| chunk_id 怎么设计？ | `(pageid << 12) \| chunk_index`，单页 < 4096 chunks，upsert 自然幂等。见 `retrieval.md` §3。 |
| pydantic-settings 怎么用？为什么不直接 os.environ？ | 自动校验类型 + `.env` 优先级处理 + `BaseSettings` 自带 IDE 提示。见 `operations.md`（如果你后面要写）。 |
| 为什么用 uv 不用 pip / poetry？ | 装依赖快 10×（Rust 写的），lockfile 跨平台稳定，自带虚拟环境。 |
| 你测过哪些 query 表现不好？ | 诚实回答："还没系统评测，但定性观察 X、Y 类型 query 召回弱，下一步打算用 RAGAS 跑 baseline"。**不要装懂没测过的事。** |

---

## Part 4 · 面试当天 cheat list

- **进门前刷一眼**：`docs/architecture.md` 的数据流图 + `docs/retrieval.md` §5（hybrid 查询步骤）。这两张图能让你 30 秒讲清整个系统。
- **白板题准备**：手写 `RateLimiter`（Q13）、手画 RRF 公式（Q2）、手写 hybrid query 伪代码（Q6）。
- **必背数字**：8000 页 / ~50k chunks / 1024 维 dense / batch 50 / RPS 3 / 30ms 查询 / 2.3GB 模型 / 提速 ~150×。
- **必答的"我不知道"**：评测数字、生产 SLA、亿级数据规模——这些**没做过就老实说**。坦诚 > 编。

---

> 最后一条：面试时**让对方翻你的 `docs/`**。一份代码 + 一套讲清 trade-off 的策略文档，是压倒性的可信度信号。
