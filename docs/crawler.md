# 爬虫策略（terraria.wiki.gg/zh）

这份文档说明 `terraria-rag` 是如何把整个中文 Terraria Wiki 拉到本地，做为 RAG 知识库的原始语料的。

目标受众：想改爬虫、或者想把这套流程套到**别的 MediaWiki 站点**的开发者。

---

## 1. 总体思路

**用 MediaWiki API，不爬渲染后的 HTML。**

terraria.wiki.gg 底层就是一套 MediaWiki 实例，每一页的"真身"都是 wikitext（`= 章节 =`、`{{模板|参数}}`、`[[链接]]` 这一套）。对 RAG 来说：

- wikitext **比 HTML 结构更稳**：不受皮肤 / JS / 模板渲染变更影响。
- 章节切分直接按 `==`、`===` 做，比解析 HTML 的 `<h2>` 干净。
- 属性表（装备数据、掉落率等）用 `{{item infobox|...}}` 这种**显式模板**表达，清洗时可以选择性保留。
- API 自带 `redirects` / `normalized` 解析，一次请求就能顺便把重定向处理好。

**代价**：拿到的是英文术语（见 README 里"已知 trade-off"里 `{{tr|...}}` 的说明）。对检索有利、对展示不利。

### 两步流水线

```
┌──────────────────┐       ┌──────────────────┐
│ 01_enumerate.py  │  ──▶  │ page_index.jsonl │  ── 所有页面的 (pageid, title)
└──────────────────┘       └──────────────────┘
                                     │
                                     ▼
┌──────────────────┐       ┌──────────────────────┐
│ 02_crawl.py      │  ──▶  │ pages/{pageid}.json  │  ── 每页一条 wikitext + 元数据
└──────────────────┘       └──────────────────────┘
```

两步分开的好处：**枚举的结果是稳定的**（变动慢），拉 wikitext 失败不影响索引；可以只"重爬失败页"而不用再问一次"有哪些页"。

---

## 2. API 接口选择

所有请求都打到：

```
GET https://terraria.wiki.gg/zh/api.php
```

带上 `format=json` + `formatversion=2`（v2 的响应结构更扁平，`pages` 是数组而不是 dict，处理起来省事）。

### 2.1 枚举页面：`list=allpages`

`scripts/01_enumerate.py` → `WikiAPIClient.iter_all_pages()`

```
action=query
list=allpages
apnamespace=0              # 主命名空间（正文）
aplimit=max                # 匿名每页最多 500 条
apfilterredir=nonredirects # 跳过重定向页，避免重复
```

- **为什么过滤重定向**：重定向页是"A 跳到 B"的 stub，本身没内容。要追 B 的内容就让 `02_crawl.py` 里的 `redirects=1` 自动解决。
- **翻页**：响应里的 `continue.apcontinue` 是游标，带回下一次请求。`iter_all_pages` 内部循环直到 `continue` 不出现为止。
- **其他命名空间**：想要 `Category:`、`File:` 等，再跑一次 `--namespace 14` / `--namespace 6` 即可（这两个 id 不是随便填的，参考 [MediaWiki 命名空间列表](https://www.mediawiki.org/wiki/Manual:Namespace)）。

产物 `data/raw/page_index.jsonl` 每行一条：

```json
{"pageid": 1234, "ns": 0, "title": "泰拉之靴"}
```

### 2.2 拉 wikitext：`prop=revisions`

`scripts/02_crawl.py` → `WikiAPIClient.fetch_wikitext_batch()`

```
action=query
prop=revisions|categories|info
titles=A|B|C|...           # 匿名单请求最多 50 个
rvprop=ids|content|timestamp
rvslots=main
cllimit=max
redirects=1
```

关键点：

- `rvslots=main` 必须带，不然 `content` 字段在 v2 结构里取不到（MediaWiki 多 slot 设计遗留）。
- `redirects=1` 让服务端自动解析重定向，响应里会多出 `redirects: [{from, to}]` 数组。
- `prop=categories` 顺便拿分类，用于后续分面检索（"武器"、"召唤"、"前哈坚石"…）。如果不用可以省掉。
- `prop=info` 留给未来用（比如想要 `length`、`lastrevid` 做 diff crawl）。

**每页落盘成 `data/raw/pages/{pageid}.json`**：

```json
{
  "pageid": 1234,
  "title": "泰拉之靴",
  "revid": 987654,
  "timestamp": "2025-08-01T12:34:56Z",
  "categories": ["Category:饰品", "Category:移动类饰品"],
  "wikitext": "{{item infobox|...}}\n'''泰拉之靴'''是..."
}
```

用 `pageid` 而不是 `title` 做文件名：title 可能含不合法的文件名字符（`/`、`:`、`?`），`pageid` 是纯数字，跨平台稳。

---

## 3. 礼貌爬取（Polite Crawling）

小 wiki 不像 Wikipedia 有大集群，我们必须克制。

### 3.1 身份识别

```
User-Agent: terraria-rag-bot/0.1 (personal study; contact: you@example.com)
```

- **必须带联系方式**：维基系站点的运维如果觉得你行为有问题，会先尝试联系，而不是直接封 IP。留个邮箱就是给自己买保险。
- wiki.gg 没有公开的 robots.txt 对 API 的特殊限制，但 API 自身就是被设计给 bot 用的——只要限速合理，不需要走 HTML。

### 3.2 限速 + 并发（`RateLimiter`）

配置项（`src/terraria_rag/config.py` / `.env`）：

| 配置 | 默认 | 含义 |
|---|---|---|
| `CRAWL_RPS` | `3.0` | 全局每秒请求数上限（所有线程共享一个 `RateLimiter`） |
| `CRAWL_CONCURRENCY` | `4` | 同时在飞的批次数（线程池 worker 数） |
| `CRAWL_BATCH_SIZE` | `50` | 单次请求里塞多少个 title（MediaWiki 匿名硬上限 = 50） |
| `CRAWL_TIMEOUT_SEC` | `30` | 单次 HTTP 请求超时 |

**并发 ≠ 绕过限速**：`RateLimiter` 是进程级的、线程安全的"最小间隔"限速器（`_next_allowed` 用 `threading.Lock` 保护，见 `api_client.py`）。并发只是让"等服务端响应"的时间能重叠，不会让出网 QPS 超过 `CRAWL_RPS`。

**默认配置（3 RPS + 4 并发 + 批量 50）对 terraria.wiki.gg 的实际压力**：
每秒 3 个请求 × 每请求 50 个 title = **~150 pages/s 的"拉取密度"**，但服务端承担的成本只是 3 req/s 的解析，这对它是小意思。对比"1 title / req × 1 req/s"的原始版本**提速 ~150×**。

### 3.3 重试策略（`tenacity`）

```python
@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
```

- 网络抖动 / 5xx / 超时 → 指数退避重试（2s → 4s → 8s → 16s → 32s，上限 60s），最多 5 次。
- **业务错误不重试**：如果 API 返回 200 + `{"missing": true}`，那是正常的"页面不存在"，不该重试；上层在 `_page_to_record` 里直接返回 `None`，记到 `failures.jsonl`。
- `reraise=True` 让最终失败原样抛出，方便 `02_crawl.py` 在 `_process_batch` 里吞掉并记录。

### 3.4 如果被 429 / 503 了

降档到：

```dotenv
CRAWL_RPS=1.5
CRAWL_CONCURRENCY=2
```

或者临时用命令行覆盖：

```bash
uv run python scripts/02_crawl.py --concurrency 2
```

指数退避会先自己顶一波，如果连续 5 次还 429 就真的需要人为降速了。

---

## 4. 断点续传与幂等性

### 4.1 设计原则

> **"跑到一半 Ctrl+C，下次再跑只补差集。"**

`02_crawl.py` 的判定逻辑：

```python
todo = [(pid, title, out) for pid, title in titles
        if not (out := pages_dir / f"{pid}.json").exists() or args.force]
```

已经落盘的 `{pageid}.json` 直接跳过，**无需读文件、无需问服务端**。全量重跑只需要 `--force`。

### 4.2 原子写

```python
tmp = out.with_suffix(out.suffix + ".tmp")
with open(tmp, "wb") as f:
    f.write(orjson.dumps(data))
tmp.replace(out)
```

先写 `.json.tmp` 再 `rename` 覆盖。`rename` 在同一文件系统上是原子的——如果进程在写到一半被杀，只会留一个 `.tmp` 垃圾文件，**不会出现"半个 JSON"**，下次跑不会因为 JSON 解析错误炸掉。

### 4.3 失败日志

`data/raw/failures.jsonl` 每行一条：

```json
{"pageid": 1234, "title": "泰拉之靴", "error": "HTTPStatusError(...)"}
```

有两类失败会进这个文件：

- **网络层**：5 次重试都没过的 `HTTPError`（整批算失败）。
- **业务层**：`{"missing": true}` 或者没 `revisions`（单页算失败，批次里其他页照常写盘）。

下次重跑 `02_crawl.py` 会自动把失败页再试一次（因为 `{pageid}.json` 不存在）；如果一直失败，把 `failures.jsonl` 拿出来人工审一下——通常是标题含特殊字符、或者页面真的被删了。

---

## 5. 边界情况处理

### 5.1 Redirects（重定向）

请求 `titles=泰拉之靴` 时，服务端可能返回：

```json
{
  "query": {
    "redirects": [{"from": "泰拉之靴", "to": "泰拉靴"}],
    "pages": [{"title": "泰拉靴", "pageid": 5678, "revisions": [...]}]
  }
}
```

`fetch_wikitext_batch` 里维护 `title_map: {原始title → 最终title}`，先过一遍 `normalized`（大小写 / 下划线规范化），再过 `redirects`，最后用 `title_map` 把结果对回调用方问的原始 title。**调用方不需要知道重定向的存在**。

### 5.2 Normalized（标题规范化）

MediaWiki 会把 `terraria_boots` 改写成 `Terraria Boots`，下划线转空格、首字母大写。响应里叫 `normalized`。处理方式同上。

### 5.3 批量请求里只有一个 title 失败

`prop=revisions` 是 best-effort 的：批里 50 个 title，有 3 个不存在，其他 47 个照样返回。我们的处理：

```python
for orig in titles:
    page = by_title.get(title_map[orig])
    out[orig] = self._page_to_record(page) if page else None
```

存在的写盘，不存在的记失败。不会因为少数坏 title 让整批报废。

### 5.4 一次请求超过 50 个 title

`fetch_wikitext_batch` 会抛 `ValueError`。`02_crawl.py` 在调用前用 `min(args.batch_size, 50)` 截断，所以正常路径触发不了，但防止有人直接调库传 100 个进来。

---

## 6. 想换一个 MediaWiki 站的话

这套爬虫是通用的，只要对方开放了 `/api.php`：

1. 改 `.env` 里的 `WIKI_BASE_URL` 和 `WIKI_LANG`（注意有的站路径不是 `/zh/api.php` 而是 `/w/api.php`——此时改 `WIKI_LANG=w` 也能蒙混过关，或者更干净地改 `api_client.py` 里的 `self.endpoint`）。
2. **必改** `WIKI_USER_AGENT`，留自己的联系方式。
3. 看对方规模降档 `CRAWL_RPS`——大站（Wikipedia）可以激进点，小 wiki 保守点。
4. 如果对方启用了 bot 账号登录（匿名上限被收紧），加一步 `action=login`——`WikiAPIClient` 目前没实现，需要补。

---

## 7. 不采用的方案 & 原因

| 方案 | 为什么没用 |
|---|---|
| 爬渲染后的 HTML | 不稳定（皮肤 / JS 变更）、解析贵、失去 wikitext 结构信息 |
| `generator=allpages` + `prop=revisions` 一步走 | 和 `continue` 游标一起用时，断点续传得自己维护 `apcontinue`；不如"枚举 → 拉内容"两步走干净 |
| `asyncio + httpx.AsyncClient` | 对这个体量（~几千页）来说，`ThreadPoolExecutor` 已经够快，asyncio 收益小、复杂度高 |
| Scrapy / Playwright | 杀鸡用牛刀，MediaWiki API 本来就是给程序访问的 |
| 不限速直接并发 50+ | 小 wiki 会被打挂或者封 IP，得不偿失 |

---

## 8. 运行参考

```bash
# 先枚举所有主命名空间页面
uv run python scripts/01_enumerate.py
# 输出：data/raw/page_index.jsonl，zh 站大概 5000~8000 行

# 再拉 wikitext，断点续传
uv run python scripts/02_crawl.py
# 默认：3 RPS × 4 并发 × 每请求 50 title，8000 页约 ~10 分钟

# 被限流了就降档
uv run python scripts/02_crawl.py --concurrency 2
# 或在 .env 里：CRAWL_RPS=1.5

# 想强制重爬（比如改了 fetch 逻辑、要覆盖旧 JSON）
uv run python scripts/02_crawl.py --force
```

成功完成后：

```
data/raw/
├── page_index.jsonl          # 枚举出的所有 (pageid, title)
├── pages/
│   ├── 1234.json             # 每页 1 个 JSON：wikitext + 元数据
│   ├── 5678.json
│   └── ...
└── failures.jsonl            # 如果有失败，人工审
```

下一步就是 `03_clean_chunk.py` —— 把 wikitext 清洗成纯文本 + 按 section 切块。另开文档说。
