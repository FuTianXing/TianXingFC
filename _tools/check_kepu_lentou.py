#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""科普文章写作铁律自检：em-dash、「不是..而是」、记账隐喻、段落长度、表格裸竖线等。

用法：python _tools/check_kepu_lentou.py <文件路径>
"""
import io
import re
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else \
    r"科普文章\借出去的分支：同一句「虚过程」，一边发散一边不发散.md"

TEXT = io.open(TARGET, encoding="utf-8").read()
LINES = TEXT.split("\n")

violations = []


def add(kind, line_no, snippet):
    violations.append((kind, line_no, snippet))


# ---------- 1. em-dash ----------
for i, ln in enumerate(LINES, 1):
    if "\u2014" in ln or "\u2013" in ln:
        add("em-dash", i, ln.strip()[:80])

# ---------- 2. 「不是..而是」及变体 ----------
PAT_NOTBUT = [
    ("不是..而是", re.compile(r"不是[^。；\n]{0,60}?而是")),
    ("并非..而是", re.compile(r"并非[^。；\n]{0,60}?而是")),
    ("不在于..而在于", re.compile(r"不在于[^。；\n]{0,60}?而在于")),
    ("而非", re.compile(r"而非")),
    ("并非", re.compile(r"并非")),
]
for i, ln in enumerate(LINES, 1):
    for name, pat in PAT_NOTBUT:
        m = pat.search(ln)
        if m:
            add(name, i, ln.strip()[:80])

# ---------- 3. 记账隐喻 ----------
LEDGER = re.compile(r"记账|账本|欠账|结账|对账|入账|出账|一笔账|两本账|三本账|漏账")
for i, ln in enumerate(LINES, 1):
    if LEDGER.search(ln):
        add("ledger", i, ln.strip()[:80])

# ---------- 4. AI 套话 ----------
BOILER = re.compile(r"首先|其次|最后|值得注意的是|众所周知|深刻|极其")
for i, ln in enumerate(LINES, 1):
    if BOILER.search(ln):
        add("boiler", i, ln.strip()[:80])

# ---------- 5. 段落长度 ----------
def is_special(s):
    t = s.strip()
    if not t:
        return True
    if t.startswith("#"):
        return True
    if t.startswith("|"):
        return True
    if t.startswith("-"):
        return True
    if t.startswith(">"):
        return True
    if t.startswith("```"):
        return True
    if t.startswith("$$"):
        return True
    if set(t) <= set("-*_= "):
        return True
    return False


para = []
para_start = 0
for i, ln in enumerate(LINES, 1):
    if is_special(ln):
        if para:
            body = "\n".join(para)
            if len(body) > 300:
                add("para>300", para_start, "%d 字符" % len(body))
            para = []
        continue
    if not para:
        para_start = i
    para.append(ln)
if para:
    body = "\n".join(para)
    if len(body) > 300:
        add("para>300", para_start, "%d 字符" % len(body))

# ---------- 6. 表格裸竖线 ----------
in_code = False
for i, ln in enumerate(LINES, 1):
    if ln.strip().startswith("```"):
        in_code = not in_code
        continue
    if in_code:
        continue
    s = ln.strip()
    if not s.startswith("|"):
        continue
    body = s
    # 表格分隔行（|---|---|）豁免
    if re.fullmatch(r"\|[\s\-:|]+\|", body):
        continue
    for m in re.finditer(r"(?<!\\)\|", body):
        pos = m.start()
        before = body[pos - 1] if pos > 0 else ""
        after = body[pos + 1] if pos + 1 < len(body) else ""
        delim = (before in ("", " ")) and (after in ("", " "))
        if not delim:
            add("table-bare-pipe", i, s[:90])

# ---------- 7. Unicode 数学符号 ----------
UNI = re.compile(r"[\u0391-\u03C9\u2200-\u22FF\u221A\u221E\u222B\u2248\u2260\u2264\u2265\u00D7\u00F7\u00B1\u221E]")
for i, ln in enumerate(LINES, 1):
    # 去掉 LaTeX 行内公式、独立公式与 wikilink 目标后检查
    stripped = re.sub(r"\$[^$]*\$", "", ln)
    stripped = re.sub(r"\[\[[^\]]+\]\]", "", stripped)
    if UNI.search(stripped):
        add("unicode-math", i, ln.strip()[:80])

# ---------- 8. LaTeX 定界符配对 ----------
for i, ln in enumerate(LINES, 1):
    if ln.count("`") % 2 != 0:
        add("backtick-odd", i, ln.strip()[:80])
    # 行内公式：排除独立 $$ 行
    if ln.strip().startswith("$$"):
        continue
    if ln.count("$") % 2 != 0:
        add("dollar-odd", i, ln.strip()[:80])

# ---------- 9. 控制字符 / 转义残部 ----------
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
for i, ln in enumerate(LINES, 1):
    if CTRL.search(ln):
        add("control-char", i, ln.strip()[:80])

# ---------- 10. wiki 词条链接（科普正文分工） ----------
for i, ln in enumerate(LINES, 1):
    for m in re.finditer(r"\[\[([^\]]+)\]\]", ln):
        tgt = m.group(1).split("|")[0].split("\\")[0].strip()
        if tgt.startswith("wiki/") or tgt.startswith("wiki\\"):
            add("wiki-link", i, tgt)

# ---------- 报告 ----------
print("target: %s" % TARGET)
print("bytes : %d" % len(TEXT.encode("utf-8")))
print("lines : %d" % len(LINES))
if violations:
    print("VIOLATIONS: %d" % len(violations))
    for kind, ln, sn in violations:
        print("  [%s] line %d: %s" % (kind, ln, sn))
    sys.exit(1)
print("ALL CLEAR")
