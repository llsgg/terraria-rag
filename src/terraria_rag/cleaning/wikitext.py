"""wikitext -> plain text + section structure.

We use mwparserfromhell to strip wiki markup, but keep:
- Section hierarchy (== heading == levels)
- Template parameters when they look like infobox/property data
- Internal/external link anchor text
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mwparserfromhell as mw
from mwparserfromhell.nodes import ExternalLink, Heading, Tag, Template, Wikilink
from mwparserfromhell.wikicode import Wikicode


# Templates with rich structured data we expand into multi-line "key: value" blocks.
# (Heuristic — names from terraria wiki + common MediaWiki conventions.)
_INFOBOX_HINTS = ("infobox", "item", "npc", "boss", "weapon", "armor", "tool", "buff", "属性")

# Templates we want to fully drop (citations, navigation, decoration that adds no semantic content).
_DROP_TEMPLATES = {
    "ref", "cite", "citation",
    "nav", "navbox", "footer",
    "stub", "cleanup", "expand", "todo",
    "exclusive", "history",
    "图标", "icon",
    "clear", "clr",
    "navbox",
}

# Inline display templates: render them as their first un-named parameter (anchor text).
# This is critical for terraria.wiki.gg where {{tr|EnglishTerm}} is the standard
# pattern for inline translations and dropping it would shred sentences.
_INLINE_TEXT_TEMPLATES = {
    "tr",           # {{tr|Aviators}} -> "Aviators"
    "lc",           # {{lc|...}} link with translation
    "l",            # generic link
    "i",            # inline italic
    "b",            # inline bold
    "rare",         # {{rare|9}} -> rarity 9
    "gametext",
    "itemtooltip",
    "tt",
}


def _tpl_name(tpl: Template) -> str:
    return str(tpl.name).strip().lower()


def _looks_like_infobox(tpl: Template) -> bool:
    name = _tpl_name(tpl)
    return any(h in name for h in _INFOBOX_HINTS)


def _first_positional(tpl: Template) -> str:
    """Return the first un-named parameter's plain text, or "" if none."""
    for param in tpl.params:
        if not str(param.name).strip().isdigit():
            continue
        val = _wikicode_to_text(param.value).strip()
        if val:
            return val
    return ""


def _flatten_template(tpl: Template) -> str:
    """Render a template as text in a way that preserves semantic content.

    Three strategies, in priority order:
    1. Drop (navigation/citation noise that adds no info)
    2. Multi-line "key: value" expansion (infoboxes / structured data)
    3. Inline text replacement: render as first positional arg, falling back to
       joining all positional args with spaces. This is the safe default — it
       preserves the *words* a sentence depends on even if we lose styling.
    """
    name = _tpl_name(tpl)
    if name in _DROP_TEMPLATES:
        return ""
    # Parser functions like {{#explode:...}}, {{#if:...}} — drop entirely;
    # we can't sensibly evaluate them client-side.
    if name.startswith("#"):
        return ""

    if _looks_like_infobox(tpl):
        lines: list[str] = [f"[{str(tpl.name).strip()}]"]
        for param in tpl.params:
            key = str(param.name).strip()
            val = _wikicode_to_text(param.value).strip()
            if not val:
                continue
            if key.isdigit():
                lines.append(f"- {val}")
            else:
                lines.append(f"- {key}: {val}")
        return "\n".join(lines) + "\n"

    # Inline templates we know about: take the first positional arg.
    if name in _INLINE_TEXT_TEMPLATES:
        return _first_positional(tpl)

    # Unknown template: try first positional. If it has none, drop.
    text = _first_positional(tpl)
    return text


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
# Final safety net: any leftover {{...}} that escaped parsing (e.g. malformed
# templates). Kill them to avoid leaking raw wiki markup into chunks.
_LEFTOVER_TPL_RE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _wikicode_to_text(code: Wikicode) -> str:
    """Recursively flatten a Wikicode object to plain text."""
    out: list[str] = []
    for node in code.nodes:
        if isinstance(node, Heading):
            out.append("\n" + "#" * node.level + " " + _wikicode_to_text(node.title).strip() + "\n")
        elif isinstance(node, Wikilink):
            # Both .title and .text may themselves contain nested templates
            # (very common on terraria.wiki.gg: [[{{tr|Master Mode}}]]).
            if node.text:
                text = _wikicode_to_text(node.text)
            else:
                text = _wikicode_to_text(node.title)
            out.append(text)
        elif isinstance(node, ExternalLink):
            # [url anchor] -> "anchor"; [url] alone -> drop
            if node.title is not None:
                out.append(_wikicode_to_text(node.title))
        elif isinstance(node, Template):
            out.append(_flatten_template(node))
        elif isinstance(node, Tag):
            tag = str(node.tag).lower()
            if tag in {"ref", "noinclude", "gallery"}:
                continue
            if node.contents is not None:
                out.append(_wikicode_to_text(node.contents))
        else:
            out.append(str(node))
    text = "".join(out)
    text = _HTML_COMMENT_RE.sub("", text)
    # Kill any leftover {{...}} that the parser failed to recognize, then
    # collapse whitespace.
    for _ in range(3):  # nested templates may need multiple passes
        new = _LEFTOVER_TPL_RE.sub("", text)
        if new == text:
            break
        text = new
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text


@dataclass
class Section:
    level: int          # 1 = top (page title), 2 = ==H==, 3 = ===H===, ...
    heading: str
    body: str           # plain text under this heading (no sub-section bodies)


def parse_to_sections(title: str, wikitext: str) -> list[Section]:
    """Split a page's wikitext into a flat list of sections, ordered by appearance.

    The first section uses the page title as its heading (level=1). Subsequent
    headings come from `== ... ==` etc.
    """
    code = mw.parse(wikitext)
    sections: list[Section] = []

    current_heading = title
    current_level = 1
    buf: list[str] = []

    def flush() -> None:
        body = "".join(buf).strip()
        if body or not sections:
            sections.append(Section(level=current_level, heading=current_heading, body=body))
        buf.clear()

    for node in code.nodes:
        if isinstance(node, Heading):
            flush()
            current_heading = _wikicode_to_text(node.title).strip()
            current_level = node.level
        else:
            sub = _wikicode_to_text(mw.wikicode.Wikicode([node]))
            if sub:
                buf.append(sub)
    flush()
    return sections
