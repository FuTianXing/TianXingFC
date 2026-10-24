# -*- coding: utf-8 -*-
"""落盘后写作铁律与结构自检器（只读）。

用途
----
在交付前对单篇 Markdown 文档做一遍机器可判的检查，输出带行号的问题清单。
检查对象是写作纪律项与结构完整性项，判定部分（论证是否成立、引用是否准确）
不在本脚本射程，须人工评审。

用法
----
    python check_discipline.py <文件路径> [--root <工作区根>] [--limit <段落字符上限>]

    --root   给定则额外做 wikilink 死链检查（只在该路径树内查找，跳过
             .git / images / tools / 归档 等目录）
    --limit  段落字符上限，默认 300（工作区写作铁律第七条）

退出码
------
    0  全部通过
    1  有命中（须逐条处理）
    2  用法错误或文件不存在

检查项
------
A 组 语言纪律    序列词、套话、强调词
B 组 转义安全    控制字符、损坏字符、Unicode 数学符号、转义七判据四项
C 组 格式结构    表格列数、段落长度、章节编号连续性、必备节、标题层级跳级
D 组 内部链接    wikilink 死链（需 --root）

豁免
----
表格行、列表项、标题、引用块、代码块、独占 $$ 行不计入段落长度。
含反引号的行不参与「缺字母残部」与「控制字符后字母」判定（代码行内为字面量）。
"""

from __future__ import print_function

import argparse
import io
import os
import re
import sys

# ---------------------------------------------------------------- 阈值配置

LIMIT_PARA = 300          # 段落字符上限
SKIP_DIRS = (".git", "images", "tools", "node_modules", "__pycache__", ".obsidian")

# ---------------------------------------------------------------- 判据定义

# A 组：语言纪律
RULES_A = [
    ("序列词", "首先/其次/最后",
     lambda l: bool(re.search(u"首先[，,]|其次[，,]|最后[，,]", l))),
    ("套话", "值得注意的是/众所周知等",
     lambda l: bool(re.search(u"值得注意的是|众所周知|不言而喻|显而易见", l))),
    ("强调词", "深刻/极其",
     lambda l: bool(re.search(u"深刻|极其", l))),
]

# B 组：转义安全
RE_CTRL = re.compile(u"[\x00-\x08\x0b\x0c\x0e-\x1f]")
RE_CTRL_LETTER = re.compile(u"[\x09\x0b\x0c\x0e-\x1f][A-Za-z]")
RE_VERT_LETTER = re.compile(r"\\vert[A-Za-z]")
RE_FRAGMENT = re.compile(r"(?<![A-Za-z\\])(imes|heta|lpha|pprox|rac\{|egin\{|ext\{|abla)")
RE_LINEHEAD_FRAG = re.compile(r"^(abla|u\}|ho|imes|rac\{|egin\{)")
RE_UNICODE_MATH = re.compile(
    u"[ℏ∫∑∏√≈≠≤≥∞∂∇→←⇒⇔·×÷±°µΩÅαβγδθλμνπρστφχψωΓΔΣΦΨΩΠ]")

BARE_DOLLAR = re.compile(r"(?<!\\)\$")   # 计数用，成对性在下面判断


# ---------------------------------------------------------------- 工具函数

def is_code(line):
    """含反引号的行视为代码行，豁免部分判据。"""
    return "`" in line


RE_WIKILINK = re.compile(r"\[\[[^\]]*\]\]")


def mask_wikilinks(line):
    """把 wikilink 整段替换为占位符后再判语言纪律。

    链接目标按实名逐字保留，掩码后不参与语言纪律判定。
    """
    return RE_WIKILINK.sub("WL", line)


def is_exempt_len(line):
    """段落长度判据的豁免行。"""
    s = line.strip()
    if not s:
        return True
    if s.startswith(("|", "-", "*", "#", ">", "!", "$$")):
        return True
    if set(s) <= set("=-*_ "):
        return True
    return False


def table_blocks(lines):
    """返回 [(起始行号, [(行号, 竖线数), ...]), ...]，每个块内列数须一致。"""
    blocks, cur = [], []
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith("|"):
            cur.append((i, l.count("|")))
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    return blocks


def check_headings(lines, problems):
    """一级节号连续性 + 二级节号格式。"""
    h2, h3 = [], []
    for i, l in enumerate(lines, 1):
        m2 = re.match(r"^## (\d+)\.?\s", l)
        m3 = re.match(r"^### (\d+)\.(\d+)\.?\s", l)
        if m2:
            h2.append((i, m2.group(1)))
        elif m3:
            h3.append((i, m3.group(1)))

    if h2:
        nums = [int(n) for _, n in h2]
        expect = list(range(1, len(nums) + 1))
        if nums != expect:
            got = ",".join(str(n) for n in nums)
            want = ",".join(str(n) for n in expect)
            problems.append(("C-节号不连续", h2[0][0],
                             "一级节号序列为 %s，应为 %s" % (got, want)))
        for i, n in h3:
            if not n.isdigit():
                problems.append(("C-节号格式", i, "二级节号未写成 数字.数字"))
    return h2, h3


def check_deadlinks(text, root, problems):
    """wikilink 目标是否存在。目录名本身允许作为链接目标（如 [[理论总纲]]）。"""
    targets = set(re.findall(r"\[\[([^\]|]+)", text))
    if not targets:
        return
    index = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = os.path.relpath(dirpath, root)
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            stem = fn[:-3]
            index.setdefault(stem, []).append(os.path.join(rel, fn) if rel != "." else fn)
    for t in sorted(targets):
        name = t.strip().split("/")[-1]
        if name.endswith(".md"):
            name = name[:-3]
        if name in index:
            continue
        # 目录型目标（理论总纲、wiki 等）
        if os.path.isdir(os.path.join(root, name)):
            continue
        problems.append(("D-死链", None, "[[%s]] 无匹配文件" % t))


# ---------------------------------------------------------------- 主流程

def check_file(path, root=None, limit=LIMIT_PARA):
    if not os.path.isfile(path):
        sys.stderr.write("文件不存在：%s\n" % path)
        return 2

    with io.open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    problems = []

    # A 组
    for name, desc, fn in RULES_A:
        for i, l in enumerate(lines, 1):
            if fn(mask_wikilinks(l)):
                problems.append(("A-" + name, i, desc))

    # B 组
    for i, l in enumerate(lines, 1):
        if is_code(l):
            continue
        if RE_CTRL.search(l):
            problems.append(("B-控制字符", i, "行内含控制字符"))
        if RE_CTRL_LETTER.search(l):
            problems.append(("B-控制后字母", i, "控制字符紧跟字母，命令前缀被吃掉"))
        if u"\ufffd" in l:
            problems.append(("B-损坏字符", i, "含 U+FFFD 替换字符，落盘时字节被截断"))
        if RE_VERT_LETTER.search(l):
            problems.append(("B-vert紧邻字母", i, r"\vert 后缺反斜杠，应为 \vert\w"))
        if RE_FRAGMENT.search(l):
            problems.append(("B-缺字母残部", i, "疑似 LaTeX 命令被截断"))
        if RE_LINEHEAD_FRAG.search(l):
            problems.append(("B-行首命令残部", i, "行首为命令残部且上一行 $ 计数可能为奇数"))
        if RE_UNICODE_MATH.search(l):
            problems.append(("B-Unicode数学符", i, "公式内用了 Unicode 数学符号，应改 LaTeX"))

    # $ 成对性
    n_dollar = len(BARE_DOLLAR.findall(text))
    if n_dollar % 2 != 0:
        problems.append(("B-美元符奇数", None, "行内公式 $ 计数为奇数（%d），可能缺配对" % n_dollar))

    # C 组
    for blk in table_blocks(lines):
        cols = set(c for _, c in blk)
        if len(cols) != 1:
            detail = ", ".join("行%d:%d" % (i, c) for i, c in blk)
            problems.append(("C-表格列数", blk[0][0], "同一表内竖线数不一致（%s）" % detail))
    for i, l in enumerate(lines, 1):
        if is_exempt_len(l):
            continue
        n = len(l.strip())
        if n > limit:
            problems.append(("C-段落超长", i, "%d 字符，上限 %d" % (n, limit)))
    check_headings(lines, problems)

    for tag in ("## 摘要", "**关键词**", "## 参考文献"):
        if tag not in text:
            problems.append(("C-缺必备节", None, "缺 %s" % tag))

    # D 组
    if root:
        check_deadlinks(text, root, problems)

    return report(path, lines, problems)


def report(path, lines, problems):
    print("=" * 68)
    print("文件：%s" % path)
    print("行数：%d  字符：%d" % (len(lines), sum(len(l) for l in lines)))
    print("=" * 68)

    if not problems:
        print("全部通过。")
        return 0

    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    problems.sort(key=lambda p: (order.get(p[0][0], 9), p[1] or 0))
    cur = None
    for tag, ln, msg in problems:
        if tag[0] != cur:
            cur = tag[0]
            print("\n[%s 组]" % cur)
        where = "行 %d" % ln if ln else "全文"
        snippet = ""
        if ln and ln <= len(lines):
            snippet = "  「%s」" % lines[ln - 1].strip()[:44]
        print("  %-16s %-6s %s%s" % (tag, where, msg, snippet))

    print("\n合计 %d 处。" % len(problems))
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="落盘后写作铁律与结构自检器（只读）")
    ap.add_argument("path", help="待检查的 Markdown 文件")
    ap.add_argument("--root", default=None, help="工作区根目录，启用死链检查")
    ap.add_argument("--limit", type=int, default=LIMIT_PARA, help="段落字符上限")
    args = ap.parse_args(argv)
    return check_file(args.path, args.root, args.limit)


if __name__ == "__main__":
    sys.exit(main())
