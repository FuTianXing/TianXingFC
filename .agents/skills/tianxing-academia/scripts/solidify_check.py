#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""solidify_check.py — 天行新学「固化进理论总纲」落盘后的统一检查。

用法：
    python solidify_check.py <文件>... [--root D:\\software\\TianXingFC] [--max-para 300]

判定分两级：

  硬伤（退出码 1）
    控制字符        写入过程把 LaTeX 命令吃成了控制字符（\\t \\v \\n 等），最快能抓出转义损坏
    路径式 wikilink 末段与文件名比对时忽略 .md 扩展名
    wikilink 未解析  裸文件名全库不存在，或路径形式的目标文件不存在

  提示（不影响退出码，供人工判断）
    超长段落        正文段落超过 --max-para 字符（表格、列表、代码、引用不计）
    序列词套话      首先/其次/最后/再次、值得注意的是/众所周知/不难看出、深刻/极其
    残余书名号      《…》引用（历史引用按规范不强制迁移，新写的应改 wikilink）

退出码 0 表示无硬伤。
"""

import argparse
import os
import re
import sys

SKIP_DIRS = ("_tools", ".git", "node_modules", "外部理论参考", "_stage0")

SOFT_PATTERNS = (
    ("序列词", re.compile(r"首先|其次|最后|再次")),
    ("套话", re.compile(r"值得注意的是|众所周知|不难看出")),
    ("强调词", re.compile(r"深刻|极其")),
)
WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def build_index(root):
    idx = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith((".md", ".py")):
                stem = os.path.splitext(name)[0]
                idx.setdefault(stem, []).append(
                    os.path.relpath(os.path.join(dirpath, name), root).replace("\\", "/"))
    return idx


def paragraphs(text):
    """产出 (起始行号, 文本) 的正文段落；跳过表格/代码/列表/引用。"""
    out, block, start = [], [], 1
    def flush(blk, st):
        if not blk:
            return
        head = blk[0].lstrip()
        if head.startswith(("|", "```", "- ", "* ", "> ")) or set(blk[0].strip()) <= set("|-: "):
            return
        out.append((st, "".join(blk)))
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip() == "":
            flush(block, start)
            block = []
        else:
            if not block:
                start = i
            block.append(line + "\n")
    flush(block, start)
    return out


def check(path, idx, max_para):
    hard, soft = [], []
    text = open(path, encoding="utf-8", newline="").read()

    bad = sorted({hex(ord(c)) for c in text if ord(c) < 0x20 and c != "\n"})
    if bad:
        hard.append(f"控制字符 {bad}（转义损坏，逐处修复）")

    stripped = WIKILINK.sub("", text)
    if "\r" in text:
        soft.append("含 CR 字符（核对行尾符策略）")

    for name, pat in SOFT_PATTERNS:
        for m in pat.finditer(stripped):
            soft.append(f"{name}：{stripped[max(0, m.start()-18):m.start()+18].strip()!r}")

    for st, para in paragraphs(text):
        flat = para.strip()
        if len(flat) > max_para:
            soft.append(f"超长段落 第 {st} 行 {len(flat)} 字符：{flat[:36]}")

    for m in WIKILINK.finditer(text):
        target = m.group(1)
        base = target.split("/")[-1]
        hits = idx.get(base)
        if not hits:
            hard.append(f"wikilink 未解析：[[{target}]]")
        elif "/" in target:
            want = os.path.splitext(target.replace("\\", "/"))[0]
            if not any(os.path.splitext(h)[0] == want or
                       os.path.splitext(h)[0].endswith("/" + want) for h in hits):
                hard.append(f"wikilink 路径形式未命中：[[{target}]]（同名在 {hits}）")
        elif len(hits) > 1:
            soft.append(f"wikilink 裸名不唯一：[[{target}]] -> {hits}")

    titles = re.findall(r"《[^》]{2,60}》", stripped)
    if titles:
        soft.append(f"残余书名号 {len(titles)} 处，例：{titles[:3]}")
    return hard, soft


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--root", default=r"D:\software\TianXingFC")
    ap.add_argument("--max-para", type=int, default=300)
    args = ap.parse_args(argv)

    idx = build_index(args.root)
    failed = False
    for path in args.files:
        hard, soft = check(path, idx, args.max_para)
        print("=" * 74)
        print(("!! " if hard else "ok ") + os.path.basename(path))
        for h in hard:
            print("   硬伤 |", h)
            failed = True
        for s in soft:
            print("   提示 |", s)
    print("=" * 74)
    print("退出码 1（有硬伤）" if failed else "退出码 0（无硬伤）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
