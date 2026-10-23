#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前检查：拒绝新增编号撞号。

背景：`验证脚本/README.md` 的编号由多条并行会话共享。占号已由
`.githooks/claim_number.py` 的锁串行化，但两条并行的 `git commit` 仍可能在
锁外撞上（一方占号后未立即提交、另一方从旧版取号）。此钩子作第二道防线：
比对 HEAD 版与暂存版的编号出现次数，只要有纯数字号被推高即拒绝。

存量重复号（2026-10-24 前的 26 处）因计数未变而不触发，符合「不追溯」的裁定。
字母后缀号（772b 等）不参与比较——它们按设计就是用来消解撞号的。
"""
import re
import subprocess
import sys

README = "验证脚本/README.md"
ENTRY = re.compile(r"^- \*\*(\d+)([a-z]*)\.")


def read_at(rev):
    """取某版本 README 的文本；rev 为 None 表示读工作区文件。"""
    if rev is None:
        try:
            with open(README, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return ""
    r = subprocess.run(["git", "show", "%s:%s" % (rev, README)],
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def counts(text):
    """统计每个纯数字号的出现次数。"""
    c = {}
    for ln in text.splitlines():
        m = ENTRY.match(ln)
        if m and not m.group(2):
            n = m.group(1)
            c[n] = c.get(n, 0) + 1
    return c


def staged_index_text():
    """读暂存区里的 README 内容（不是工作区版本）。"""
    r = subprocess.run(["git", "show", ":" + README],
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def main():
    head = read_at("HEAD")
    staged = staged_index_text()
    if not staged:
        return 0

    h, s = counts(head), counts(staged)
    new_dups = sorted([n for n, v in s.items() if v > 1 and v > h.get(n, 0)],
                      key=lambda x: int(x))
    if not new_dups:
        return 0

    print("=" * 68)
    print("提交被拒绝：本次提交新增了编号撞号")
    print("=" * 68)
    for n in new_dups:
        print("  第 %s 号：HEAD 版 %d 条 -> 暂存版 %d 条"
              % (n, h.get(n, 0), s[n]))
    print("")
    print("处置（AGENTS.md「README 编号占位规范」第 3 条）：")
    print("  撞号不顺延。先落盘者保留原号，后落盘者加字母后缀 b、c、d。")
    print("  后缀须同步三处：条目行 **<号><后缀>.**、注记标题与文件名、全部引用处。")
    print("")
    print("  占号请走：python .githooks/claim_number.py <本线名>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
