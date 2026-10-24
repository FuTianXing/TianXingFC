#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""archive_audit.py — 天行新学「作废轮次归档」的四段核验。

用法：
    python archive_audit.py --archive 归档/洛伦兹线误设构型轮次 \\
                            --old 验证脚本 --old 内部草稿 \\
                            [--root D:\\software\\TianXingFC]

四段（对应 SKILL.md 第 6 节）：

  A 残留      全库是否还有链接指向旧路径（旧目录下的归档件名）。期望 0。
  B 出链      归档目录内每个 md 的 wikilink 与 ![]() 是否全部解析。期望全部解析。
  C 入链      全库指向归档目录的 wikilink 目标是否都存在。期望全部存在。
  D 暂存      原路径的删除是否都已进暂存区。期望无遗留 D。

退出码 0 表示四段全过，1 表示至少一段有残留。

扫描范围排除：.git、_tools、node_modules、外部理论参考，以及 agent 记忆目录
（.trae、.workbuddy、.workbuddy-ai、.zcode、.claude、.reasonix、.agents）。
这些目录不在 AGENTS.md 目录规范内，把它们算进来会报大量与工作区无关的死链。
"""

import argparse
import os
import subprocess
import sys

SKIP_DIRS = {
    ".git", "_tools", "node_modules", "外部理论参考",
    ".trae", ".workbuddy", ".workbuddy-ai", ".zcode", ".claude",
    ".reasonix", ".agents", ".githooks", "__pycache__",
}


def norm(p):
    return p.replace(os.sep, "/")


def iter_md(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def build_index(root):
    """全库 md 的 basename 与相对路径两套索引，供 wikilink 解析。"""
    base, rel = set(), set()
    for path in iter_md(root):
        r = norm(os.path.relpath(path, root))
        rel.add(r)
        rel.add(r[:-3] if r.endswith(".md") else r)
        base.add(os.path.basename(r)[:-3])
    return base, rel


def parse_links(text):
    """返回 (wikilink 目标列表, 图片/普通 markdown 链接路径列表)。"""
    wikis, md_links = [], []
    i = 0
    while True:
        j = text.find("[[", i)
        if j < 0:
            break
        k = text.find("]]", j + 2)
        if k < 0:
            break
        target = text[j + 2:k].split("|")[0].strip()
        if target:
            wikis.append(target)
        i = k + 2
    i = 0
    while True:
        j = text.find("](", i)
        if j < 0:
            break
        k = text.find(")", j + 2)
        if k < 0:
            break
        target = text[j + 2:k].strip()
        if target and not target.startswith(("http://", "https://", "file://", "#")):
            md_links.append(target)
        i = k + 1
    return wikis, md_links


def resolve_wiki(target, base, rel):
    t = target.strip().rstrip("\\")
    if t.endswith(".md"):
        t = t[:-3]
    if "/" in t:
        return t.lstrip("./") in rel or (t + ".md").lstrip("./") in rel
    return t in base


def audit_a(root, archive_rel, old_dirs, base, rel, moved_names=None):
    """A 段：是否还有链接指向旧路径下的归档件。

    moved 名单默认取归档目录下的 md，但**必须排除 README**：归档目录的
    README 是归档时新建的说明文件，从未在 `验证脚本/` 或 `内部草稿/` 下存在过，
    把它算进名单会把全库所有 `[[验证脚本/README]]` 误报成本轮残留。
    必要时用 --moved 给显式名单，避免任何猜测。
    """
    if moved_names:
        moved = set(moved_names)
    else:
        moved = set()
        for fn in os.listdir(os.path.join(root, archive_rel)):
            if fn.endswith(".md") and fn.lower() != "readme.md":
                moved.add(fn[:-3])
    hits = []
    for path in iter_md(root):
        r = norm(os.path.relpath(path, root))
        if r.startswith(archive_rel + "/"):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        wikis, _ = parse_links(text)
        for target in wikis:
            t = target.strip().rstrip("\\")
            if t.endswith(".md"):
                t = t[:-3]
            for od in old_dirs:
                if t.startswith(od + "/") and os.path.basename(t) in moved:
                    hits.append((r, target))
    return hits


def audit_b(root, archive_rel, base, rel):
    """B 段：归档目录内每个 md 的出链是否全部解析。"""
    bad = []
    adir = os.path.join(root, archive_rel)
    for path in iter_md(adir):
        r = norm(os.path.relpath(path, root))
        text = open(path, encoding="utf-8", errors="ignore").read()
        wikis, md_links = parse_links(text)
        for target in wikis:
            if not resolve_wiki(target, base, rel):
                bad.append((r, "wikilink", target))
        for target in md_links:
            t = target.split("#")[0].strip()
            if not t:
                continue
            full = os.path.normpath(os.path.join(os.path.dirname(path), t))
            if not os.path.exists(full):
                bad.append((r, "path", target))
    return bad


def audit_c(root, archive_rel, base, rel):
    """C 段：全库指向归档目录的 wikilink 目标是否都存在。"""
    bad = []
    for path in iter_md(root):
        r = norm(os.path.relpath(path, root))
        text = open(path, encoding="utf-8", errors="ignore").read()
        wikis, _ = parse_links(text)
        for target in wikis:
            t = target.strip().rstrip("\\")
            if t.endswith(".md"):
                t = t[:-3]
            if t.startswith(archive_rel + "/") and not resolve_wiki(target, base, rel):
                bad.append((r, target))
    return bad


def audit_d(root):
    """D 段：原路径的删除是否都已进暂存区。"""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
        ).stdout
    except OSError:
        return []
    pending = []
    for line in out.splitlines():
        if len(line) > 3 and line[:2] in ("D ", " D") and not line.startswith("D  "):
            pending.append(line[3:])
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="归档目录，相对工作区根")
    ap.add_argument("--old", action="append", default=[],
                    help="原目录，可多次给出（如 验证脚本 内部草稿）")
    ap.add_argument("--root", default=r"D:\software\TianXingFC")
    ap.add_argument("--moved", action="append", default=[],
                    help="显式给出已移动的 md 名（不含 .md），可多次；"
                         "不给时取归档目录内除 README 外的 md")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    archive_rel = norm(args.archive).strip("/")
    if not os.path.isdir(os.path.join(root, archive_rel)):
        print(f"归档目录不存在：{archive_rel}")
        return 2

    base, rel = build_index(root)
    a = audit_a(root, archive_rel, args.old, base, rel, args.moved or None)
    b = audit_b(root, archive_rel, base, rel)
    c = audit_c(root, archive_rel, base, rel)
    d = audit_d(root)

    def report(tag, title, items, ok_msg):
        print(f"[{tag}] {title}：{len(items)}")
        for item in items[:40]:
            print("     ", item)
        if len(items) > 40:
            print(f"      …… 另有 {len(items) - 40} 条")
        if not items:
            print(f"      {ok_msg}")

    report("A", "残留（指向旧路径的链接）", a, "无残留")
    report("B", "出链（归档目录内未解析）", b, "全部解析")
    report("C", "入链（指向归档目录的未解析目标）", c, "全部存在")
    report("D", "暂存（原路径未暂存的删除）", d, "无遗留")

    bad = len(a) + len(b) + len(c) + len(d)
    print()
    print("四段全过。" if bad == 0 else f"共 {bad} 处待处理。")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
