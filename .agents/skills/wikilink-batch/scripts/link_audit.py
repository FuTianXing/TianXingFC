#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wikilink-batch 审计器。

对目标 md 做五类检查：
  1. wikilink 目标解析（裸名全库唯一 / 路径形式存在）
  2. 剩余书名号引用清单（--suggest 附候选文件与重名告警）
  3. wikilink 外的中文 em-dash（文件名内的合法）
  4. 写作铁律禁词（不是……而是及变体、值得注意、众所周知、账）
  5. 编号条目缺链清单与 CRLF 残留

退出码：0 = 无硬伤；1 = 有硬伤（未解析目标 / em-dash / 禁词）。
剩余《》与缺链条目为 INFO 级，不计入退出码。
"""
import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

BANNED = [
    "不是……而是", "不是...而是",
    "并非……而是", "并非...而是",
    "不在于……而在于", "不在于...而在于",
    "值得注意的是", "众所周知", "账",
]


def find_vault_root(start):
    """从文件位置向上找库根（AGENTS.md 或 .git 所在目录）。"""
    d = os.path.dirname(os.path.abspath(start))
    while True:
        if os.path.exists(os.path.join(d, "AGENTS.md")) or os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def vault_md_index(root):
    """库内全部 md：basename -> [相对路径(不含.md)]。"""
    index = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in (".git", "images", "node_modules")]
        for fn in filenames:
            if fn.endswith(".md"):
                rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")
                index.setdefault(fn[:-3], []).append(rel[:-3])
    return index


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def audit_file(path, root, suggest):
    issues = []
    infos = []
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    lines = text.splitlines()
    stripped = re.sub(r"\[\[[^\]]*\]\]", "\uFFFD", text)  # wikilink 整体屏蔽后做文本级扫描
    index = vault_md_index(root) if root else {}

    # 1. wikilink 目标解析
    unresolved = []
    seen = set()
    for m in re.finditer(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", text):
        t = m.group(1).strip()
        if t in seen:
            continue
        seen.add(t)
        if "/" in t:
            if not os.path.exists(os.path.join(root, t + ".md")):
                unresolved.append((t, line_of(text, m.start()), "路径不存在"))
        else:
            c = len(index.get(t, [])) if root else -1
            if c != 1:
                unresolved.append((t, line_of(text, m.start()), "命中 %d 个同名文件" % c))
    if unresolved:
        issues.append(("目标未解析", ["行 %d  %s（%s）" % (ln, t, why) for t, ln, why in unresolved]))

    # 2. 剩余书名号（按唯一引用归组，行号最多列 4 个）
    cites = []
    for m in re.finditer(r"《[^》]*》", text):
        cites.append((m.group(0), line_of(text, m.start())))
    if cites:
        by_cite = {}
        for c, ln in cites:
            by_cite.setdefault(c, []).append(ln)
        lines_out = [
            "%s @ 行 %s" % (c, ",".join(map(str, lns[:4])) + ("…" if len(lns) > 4 else ""))
            for c, lns in sorted(by_cite.items())
        ]
        if suggest and root:
            uniq = sorted(set(c for c, _ in cites))

            def norm(s):
                return re.sub(r"[：:，,、。·\s\-—–_（）()]", "", s)

            norm_index = {}
            for b, rels in index.items():
                norm_index.setdefault(norm(b), []).extend(rels)
            cand_out = []
            for c in uniq:
                bare = c[1:-1]
                exact = index.get(bare, [])
                if exact:
                    tag = "精确[%d]" % len(exact) if len(exact) == 1 else "重名![%d]" % len(exact)
                    cand_out.append("  %s -> %s %s" % (c, tag, "; ".join(exact)))
                    continue
                nb = norm(bare)
                eq = norm_index.get(nb, [])
                if eq:
                    uniq_r = sorted(set(eq))
                    tag = "近似[%d]" % len(uniq_r) if len(uniq_r) == 1 else "近似重名![%d]" % len(uniq_r)
                    cand_out.append("  %s -> %s %s" % (c, tag, "; ".join(uniq_r)))
                    continue
                fwd, rev = [], []
                for key, rels in norm_index.items():
                    if key != nb and key.startswith(nb):
                        fwd.extend(rels)
                    elif key != nb and nb.startswith(key):
                        rev.extend(rels)
                if fwd:
                    uniq_r = sorted(set(fwd))
                    tag = "短名候选[%d]" % len(uniq_r) if len(uniq_r) == 1 else "短名候选多义![%d]" % len(uniq_r)
                    cand_out.append("  %s -> %s %s" % (c, tag, "; ".join(uniq_r[:4])))
                elif rev:
                    uniq_r = sorted(set(rev))
                    tag = "弱候选[%d]" % len(uniq_r) if len(uniq_r) == 1 else "弱候选多义![%d]" % len(uniq_r)
                    cand_out.append("  %s -> %s %s" % (c, tag, "; ".join(uniq_r[:4])))
                else:
                    cand_out.append("  %s -> 无候选" % c)
            infos.append(("书名号候选解析", lines_out + cand_out))
        else:
            infos.append(("剩余书名号（INFO，历史文档不强制迁移）", lines_out))

    # 3. em-dash（wikilink 外）
    dash = []
    for i, ln in enumerate(lines, 1):
        s = re.sub(r"\[\[[^\]]*\]\]", "\uFFFD", ln)
        if "——" in s or "—" in s:
            dash.append(i)
    if dash:
        issues.append(("wikilink 外 em-dash", ["行 %d" % i for i in dash]))

    # 4. 禁词
    banned_hits = []
    for b in BANNED:
        if b in stripped:
            hits = [line_of(stripped, m.start()) for m in re.finditer(re.escape(b), stripped)]
            banned_hits.append("%s @ 行 %s" % (b, ",".join(map(str, hits[:5]))))
    if banned_hits:
        issues.append(("禁词命中", banned_hits))

    # 5. 编号条目缺链（INFO）
    no_link = []
    for i, ln in enumerate(lines, 1):
        if re.match(r"^\d+\. ", ln) and "[[" not in ln:
            no_link.append("行 %d  %s" % (i, ln[:30]))
    if no_link:
        infos.append(("编号条目缺链（INFO）", no_link))

    # 6. CRLF
    crlf = text.count("\r\n")
    if crlf:
        infos.append(("CRLF 残留", ["%d 处" % crlf]))

    return issues, infos, len(lines)


def main():
    ap = argparse.ArgumentParser(description="wikilink-batch 审计器")
    ap.add_argument("files", nargs="+", help="待审计 md 文件")
    ap.add_argument("--root", default=None, help="库根目录（默认从文件位置向上探测）")
    ap.add_argument("--suggest", action="store_true", help="为剩余书名号给出候选文件")
    args = ap.parse_args()

    exit_code = 0
    for path in args.files:
        path = os.path.abspath(path)
        root = args.root or find_vault_root(path)
        if not os.path.exists(path):
            print("文件不存在: %s" % path)
            exit_code = 1
            continue
        if not root:
            print("无法定位库根（无 AGENTS.md/.git），目标解析跳过: %s" % path)
            root = None
        issues, infos, nlines = audit_file(path, root, args.suggest)
        status = "PASS" if not issues else "FAIL"
        print("== %s  %s  (%d 行, 库根: %s)" % (status, os.path.basename(path), nlines, root))
        for name, items in issues:
            print("  [硬伤] %s:" % name)
            for it in items:
                print("    " + it)
        for name, items in infos:
            print("  [info] %s:" % name)
            for it in items:
                print("    " + it)
        if issues:
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
