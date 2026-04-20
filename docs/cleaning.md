# Wikitext 清洗策略

把 `data/raw/pages/{pageid}.json` 里的 wikitext 转成可以喂给 embedding 模型的纯文本，**同时保留对检索有用的结构信息**。

代码：`src/terraria_rag/cleaning/wikitext.py`，被 `scripts/03_clean_chunk.py` 调用。

> 前置阅读：[`architecture.md`](./architecture.md)（看你大概在流水线哪一段）。

---

## 1. 为什么不能直接 strip 掉所有 markup

Wiki 的 markup 不是装饰，它**承载语义**：

| Wikitext 片段 | 直接删掉的后果 |
|---|---|
| `== 制作 ==` | 丢失章节边界，整篇变一坨 |
| `{{tr|Master Mode}}` | 句子里直接消失一个名词，"在 中可获得" |
| `{{item infobox \| 攻击=20 \| 击退=4}}` | 装备数据全没，问"伤害多少"无法回答 |
| `[[泰拉之靴|靴子]]` | 显示文本"靴子"丢，留下原标题"泰拉之靴" |
| `<ref>来源</ref>` | 引用噪声，应该删 |
| `<!-- 编辑提示 -->` | HTML 注释，应该删 |

所以清洗策略不是"删 markup"，而是**逐种语法分类处理**。

---

## 2. 总体流程

```
wikitext (str)
   │
   ▼ mwparserfromhell.parse(..)
Wikicode (AST)
   │
   ▼ 递归遍历，每种节点类型分别处理
plain text (str)
   │
   ▼ parse_to_sections(title, wikitext)
list[Section{level, heading, body}]
```

`mwparserfromhell` 是 Python 生态里最成熟的 wikitext 解析器（Wikipedia 官方工具链在用）。我们不自己 regex 解析的原因：

- wikitext 嵌套地狱：`[[{{tr|Master Mode}}|主模式]]` 里 `[[]]` 套 `{{}}`，正则做不动。
- 模板参数分隔符 `|` 在 `[[]]` 里有歧义。
- HTML 标签、注释、parser functions（`{{#if:...}}`）一并要处理。

---

## 3. 节点类型处理表

`_wikicode_to_text` 是核心递归函数，按节点类型分发：

| 节点类型 | 处理方式 | 例子 |
|---|---|---|
| `Heading` | 转成 `# / ## / ###` 形式（保留层级） | `== 制作 ==` → `\n## 制作\n` |
| `Wikilink` | 取显示文本（`text` 优先，否则 `title`），递归处理（因为里面可能套模板） | `[[泰拉之靴\|靴子]]` → `靴子` |
| `ExternalLink` | 取 anchor，没 anchor 就丢 | `[https://x.com 官网]` → `官网` |
| `Template` | **见第 4 节**，三种策略 | `{{tr|Aviators}}` → `Aviators` |
| `Tag` | `<ref>` `<noinclude>` `<gallery>` 全删；其他取内容 | `<ref>来源</ref>` → `` |
| 其他（Text 节点等） | 原样保留 | |

最后做三件清理：

1. 删 HTML 注释 `<!--...-->`。
2. **多 pass** 删残留 `{{...}}`（解析失败的、嵌套深的，最多 3 轮）。
3. 折叠空白 + 折叠 3+ 空行为 2 空行。

---

## 4. 模板（`{{...}}`）的三种策略

模板是 wikitext 里**最复杂**的部分，没有"一招通吃"。我们把模板分三类：

### 4.1 直接丢（`_DROP_TEMPLATES`）

无语义价值的导航 / 引用 / 维护标签：

```python
_DROP_TEMPLATES = {
    "ref", "cite", "citation",
    "nav", "navbox", "footer",
    "stub", "cleanup", "expand", "todo",
    "exclusive", "history",
    "图标", "icon",
    "clear", "clr",
}
```

加上 parser functions（`{{#if:}}`、`{{#switch:}}` 这种以 `#` 开头的）也全丢——客户端没办法在不渲染整个 wiki 的情况下求值。

### 4.2 展开成 "key: value" 块（infobox / 结构化数据）

判断条件：模板名里含任一关键词

```python
_INFOBOX_HINTS = ("infobox", "item", "npc", "boss",
                  "weapon", "armor", "tool", "buff", "属性")
```

这些模板里一般是**装备数据 / NPC 属性 / Boss 数值**，对问答系统是黄金信号。我们展开成多行 key:value，让模型能直接看到字段名和值：

输入：

```
{{item infobox
| 名称   = 飞行员风镜
| 类型   = 饰品
| 防御力 = 0
| 击退   = 0
| 稀有度 = 9
}}
```

输出：

```
[item infobox]
- 名称: 飞行员风镜
- 类型: 饰品
- 防御力: 0
- 击退: 0
- 稀有度: 9
```

带方括号头部 `[item infobox]` 的好处：embedding 时这个标记就是个"这里是结构化数据"的弱信号；BM25 那边匹配 "infobox" 也能命中。

> ⚠️ 注意：模板名识别是**子串匹配**（`"infobox" in name`），所以 `npc infobox`、`item infobox`、`boss infobox` 都会命中。但也意味着 `{{infoboxnav}}` 这种假名字会误中——目前 terraria.wiki.gg 上没遇到，遇到了再加 deny list。

### 4.3 取首参数（inline templates）

```python
_INLINE_TEXT_TEMPLATES = {
    "tr",      # {{tr|Aviators}} -> "Aviators"
    "lc", "l",
    "i", "b",
    "rare",    # {{rare|9}} -> "9"
    "gametext",
    "itemtooltip", "tt",
}
```

这些模板在原文里是**句子的一部分**，必须保留为内联文本。最关键的是 `{{tr|...}}`：terraria.wiki.gg 中文站到处是 `在{{tr|Hardmode}}中可以获得`，把它们整个删掉句子就破了。

策略：取第一个数字参数（positional arg）的纯文本。`{{tr|Master Mode}}` → `Master Mode`，`{{rare|9}}` → `9`。

### 4.4 未知模板：fallback 到首参数

不在 drop / infobox / inline 里的模板，**优先级最低、最保守**地处理：

```python
text = _first_positional(tpl)
return text  # 没有 positional 就返回 ""
```

这样做的逻辑：未知模板里如果有人类可读的"主参数"，那它大概率就是显示文本，留下来不亏；如果连首参数都没有，多半是纯渲染指令（如 `{{clear}}`），删了也无害。

---

## 5. Section 切分

`parse_to_sections(title, wikitext)` 输出 `list[Section]`：

```python
@dataclass
class Section:
    level: int          # 1=页标题, 2==H==, 3===H===, ...
    heading: str
    body: str           # 该 heading 下的纯文本（不含子 heading 的 body）
```

**关键设计**：每一个 heading 单独成 section，**子 heading 不归属父 section**。例：

```
泰拉之靴                    ← Section(level=1, heading="泰拉之靴")
  这是一双饰品。
== 制作 ==                  ← Section(level=2, heading="制作")
  在工作台合成。
=== 配方 ===                ← Section(level=3, heading="配方")
  - 火箭靴 + ...
== 笔记 ==                  ← Section(level=2, heading="笔记")
  ...
```

输出 4 个 Section，**body 不嵌套**。"配方"的内容只在 `level=3` 的那个 Section 里，不会同时出现在 "制作" 的 body 里。

为什么这样：嵌套会导致同一段文本被切到多个 chunk，**重复入库**，既占存储又拉低 RRF 分数。把"层级关系"留给下游（`chunking/splitter.py` 里的 `_build_section_paths` 会重建 breadcrumb）。

---

## 6. 已知限制

| 问题 | 现状 | 影响 |
|---|---|---|
| `<gallery>` 标签整个丢 | 故意的 | 失去图片 caption 的文本，但 gallery 里的文字一般是文件名，价值低 |
| Lua 模块 `{{#invoke:...}}` | 当作 parser function 整个丢 | terraria.wiki.gg 上少见，丢了不疼 |
| `{{tr}}` 的中文映射 | 只取英文原文 | RAG 检索友好，但展示要做术语后处理（见 [`api.md`](./api.md)） |
| Table（`{| ... |}`） | mwparserfromhell 默认转成纯文本 | 表格语义会扁平化（行列关系丢失），但 cells 内文字保留 |
| 模板嵌套 > 3 层 | leftover regex 只跑 3 轮 | 极端 corner case 可能漏 `{{` 残留；目前没观察到 |

如果你发现某个页面清洗后明显丢信息，最快的诊断：

```python
from terraria_rag.cleaning.wikitext import parse_to_sections
import orjson

p = orjson.loads(open("data/raw/pages/1234.json","rb").read())
for s in parse_to_sections(p["title"], p["wikitext"]):
    print("="*40, s.level, s.heading)
    print(s.body)
```

---

## 7. 不采用的方案

| 方案 | 为什么没用 |
|---|---|
| 用 `mwparserfromhell.strip_code()` | 太粗暴，模板参数和 `{{tr}}` 全丢，infobox 数据等于没了 |
| 走 `action=parse` 拿 HTML 再清洗 | 多一次网络 + 失去模板原始结构 + 解析 HTML 复杂度更高 |
| 写自己的正则 wikitext parser | 嵌套 / 转义 / parser functions 处理不动；造轮子还不一定比 mwparserfromhell 好 |
| 保留 wikitext 原文喂给 embedding | BGE-M3 没专门训过 wikitext 语法，`{{ }}` 这种符号会污染 attention |

---

## 8. 改这一层的注意事项

- **模板列表是经验性的**。换站点（如 minecraft.wiki）时 `_INFOBOX_HINTS` / `_INLINE_TEXT_TEMPLATES` / `_DROP_TEMPLATES` 都要重新审一遍。
- 改完后**必须重跑 `03_clean_chunk.py` + `04_index.py --rebuild`**——chunks 文本变了，向量也得重算。
- 加新模板分类时，在 `_INLINE_TEXT_TEMPLATES` 里加一条比改 `_flatten_template` 函数体安全得多。

---

下一步：[`chunking.md`](./chunking.md)（怎么把 sections 切成 chunks）
