#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wikilink-batch 批量转换器：书名号引用与参考文献条目转 wikilink。

实战修正版（2026-09-12，见 SKILL.md 踩坑清单）：
  1. 裸名命中必须 basename 全库计数 == 1（根目录文件 relpath == 裸名，
     直接查 relpath 集合会绕过重名检查，曾写出 [[离散镶嵌：…]] 歧义裸链）；
  2. 短名前缀匹配排除报告类文件（评审/审查/总结/复审报告），其文件名
     以正稿题开头会污染候选；正文类候选存在但重名时宁可不链；
  3. 归一化匹配：分隔符统一（——/_/-/（/，→：，去空格）+ 宽松形
     （只留中文与字母数字，吸收 $Z_3$/Z₃、10^{-17}/10^-17 差异）；
  4. 剥括注、[M]类标注、注记：类前缀、冒号前段；前缀补全试
     注记-/论文草案-/研究方向-/总结报告-/评审报告-；
  5. span 级幂等处理：不整文件跳过"已含 [[ "的文件（会漏转残留引用）；
  6. 修订记录区、标题行、代码围栏不动；已转换 span 不重入；
  7. 文献条目用零宽断言保留 ". " 分隔符（曾吞分隔符 146 处）。

用法：
  python batch_link.py 文件1.md 文件2.md ... [--root 库根]
  python batch_link.py --dir 内部草稿 --prefix "注记-","研究方向-" [--root 库根]

退出码 0 = 无改动或全部落盘成功；2 = 有锚点断言失败（不落盘）。
"""
import argparse
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SKIP_DIRS = {".git", "node_modules", "_tmp", "_trash_20260911", "_stage0"}
SPAN = re.compile(r"《([^《》]{1,120}?)》")
REF_LINE = re.compile(
    r"^(\[\d+\]\s*傅天行\.?\s*(?:\(\d{4}\)\s*\.?\s*|（\d{4}）\s*\.?\s*)?)(.+?)\.\s*(?=天行新学)")
PREFIXES = ("注记-", "论文草案-", "研究方向-", "总结报告-", "评审报告-")
REPORT_WORDS = ("评审报告", "审查报告", "总结报告", "复审报告", "提示词")
LOOSE = re.compile(r"[^\w\u4e00-\u9fff]+")
CAT = re.compile(r"^(注记|研究方向|论文草案|总结报告|评审报告|内部草稿)[：:-]+(.+)$")


def nsq(s):
    """归一化：分隔符统一为全角冒号，去空格。"""
    for ch in ("——", "_", "-", "（", "，"):
        s = s.replace(ch, "：")
    return s.replace("）", "").replace(" ", "").replace("　", "")


def loose(s):
    """宽松形：只留中文与字母数字（吸收 $...$、^、{}、下标字符差异）。"""
    return LOOSE.sub("", s.replace("₃", "3").replace("²", "2"))


def expand_forms(bare):
    """剥离括注、§节引、[M]类标注、类别前缀、冒号前段，迭代到不动点。"""
    forms, seen, i = [bare], {bare}, 0
    while i < len(forms):
        f = forms[i]
        i += 1
        cands = [
            re.sub(r"（[^（]*）", "", f).strip(),
            re.sub(r"\s*§.*$", "", f).strip(),
            re.sub(r"\s*\[[A-Z/]+\]\s*$", "", f).strip(),
        ]
        m = CAT.match(f)
        if m:
            cands.append(m.group(2).strip())
        if "：" in f:
            cands.append(f.split("：")[0].strip())
        for c in cands:
            if c and c not in seen:
                seen.add(c)
                forms.append(c)
    return forms


class Linker:
    def __init__(self, root):
        self.root = root
        rel_paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(".md"):
                    rel = os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/")[:-3]
                    rel_paths.append(rel)
        self.name_counts = Counter(p.split("/")[-1] for p in rel_paths)
        self.rel_set = set(rel_paths)
        self.unique_bare = {n for n, c in self.name_counts.items() if c == 1}
        self.nsq_index = self._uniq_index(nsq)
        self.L_index = self._uniq_index(loose)

    def _uniq_index(self, key):
        idx, counts = {}, Counter()
        for n in self.unique_bare:
            idx.setdefault(key(n), n)
            counts[key(n)] += 1
        return {k: v for k, v in idx.items() if counts[k] == 1}

    def _prefix_match(self, key):
        """短名前缀候选：正文类优先；正文类重名宁可不链；仅报告类命中且唯一才取。"""
        body = [n for n in self.name_counts
                if nsq(n).startswith(key) and len(nsq(n)) > len(key)
                and nsq(n)[len(key)] == "："
                and all(t not in n for t in REPORT_WORDS)]
        uniq = [n for n in body if self.name_counts[n] == 1]
        if body:
            return uniq[0] if len(uniq) == 1 else None
        rep = [n for n in self.name_counts
               if nsq(n).startswith(key) and len(nsq(n)) > len(key)
               and nsq(n)[len(key)] == "：" and self.name_counts[n] == 1
               and any(t in n for t in REPORT_WORDS)]
        return min(rep, key=len) if len(rep) == 1 else None

    def _try(self, f, inner):
        if f in self.unique_bare:
            return "[[" + f + "]]" if f == inner else "[[" + f + "|" + inner + "]]"
        k = nsq(f)
        if k and k in self.nsq_index:
            return "[[" + self.nsq_index[k] + "|" + inner + "]]"
        if len(k) >= 4:
            t = self._prefix_match(k)
            if t:
                return "[[" + t + "|" + inner + "]]"
            for pfx in PREFIXES:
                t = self._prefix_match(nsq(pfx) + k)
                if t:
                    return "[[" + t + "|" + inner + "]]"
        lk = loose(f)
        if len(lk) >= 4 and lk in self.L_index:
            return "[[" + self.L_index[lk] + "|" + inner + "]]"
        return None

    def resolve(self, inner):
        """书名号内文/文献标题 -> 链接文本或 None（宁可不链）。"""
        inner = inner.strip()
        if "/" in inner and inner in self.rel_set:
            return "[[" + inner + "]]"
        if "/" not in inner and self.name_counts.get(inner) == 1:
            return "[[" + inner + "]]"
        bare = inner.split("/")[-1].strip() if "/" in inner else inner
        if bare.endswith(".md"):
            bare = bare[:-3].strip()
        for f in expand_forms(bare):
            link = self._try(f, inner)
            if link:
                return link
        key = nsq(bare)
        if len(key) >= 8:
            cands = [n for n in self.unique_bare
                     if nsq(n).endswith(key) and nsq(n) != key]
            if len(cands) == 1:
                return "[[" + cands[0] + "|" + inner + "]]"
        if len(bare) >= 5:
            cands = [n for n in self.unique_bare
                     if bare.endswith(n) and len(bare) > len(n) and len(n) >= 3]
            if len(cands) == 1:
                return "[[" + cands[0] + "|" + inner + "]]"
        return None


def convert_file(path, linker):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    m = re.search(r"^(?:#{1,6} |- \*\*|\*\*)修订记录(?:\*\*)?\s*$", text, re.M)
    cut = m.start() if m else len(text)
    head, tail = text[:cut], text[cut:]

    in_fence = False
    n_link, n_ref, unresolved = 0, 0, []
    out_lines = []
    for line in head.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence or line.startswith("#"):
            out_lines.append(line)
            continue
        if re.match(r"^\[\d+\] ", line):
            # 文献条目：前瞻保留 ". " 分隔符；已含链接的行跳过
            if "[[" not in line:
                m = REF_LINE.match(line)
                if m:
                    link = linker.resolve(m.group(2))
                    if link:
                        line = m.group(1) + link + ". " + line[m.end():]
                        n_ref += 1
                    else:
                        unresolved.append("文献:" + m.group(2))
            out_lines.append(line)
            continue
        pos, pieces = 0, []
        for sm in SPAN.finditer(line):
            if line[:sm.start()].count("[[") > line[:sm.start()].count("]]"):
                continue  # 已在既有链接内
            link = linker.resolve(sm.group(1))
            if link:
                pieces.append(line[pos:sm.start()])
                pieces.append(link)
                pos = sm.end()
                n_link += 1
            else:
                unresolved.append(sm.group(1))
        pieces.append(line[pos:])
        out_lines.append("".join(pieces))

    new_text = "\n".join(out_lines) + tail
    if new_text == text:
        return None, n_link, n_ref, unresolved
    assert new_text.count("\n") == text.count("\n"), path + " 行数变化"
    assert new_text.count("\r\n") == text.count("\r\n"), path + " CRLF 变化"
    assert new_text.endswith(tail), path + " 修订记录区被改动"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)
    return True, n_link, n_ref, unresolved


def main():
    ap = argparse.ArgumentParser(description="wikilink-batch 批量转换器")
    ap.add_argument("files", nargs="*", help="待处理 md 文件")
    ap.add_argument("--dir", help="目录模式：处理该目录下匹配前缀的 md")
    ap.add_argument("--prefix", default="", help="目录模式的文件名前缀（逗号分隔）")
    ap.add_argument("--skip", default="", help="强制跳过的文件名（逗号分隔，如在途文件）")
    ap.add_argument("--root", default=None, help="库根目录（默认从文件位置向上探测）")
    args = ap.parse_args()

    files = list(args.files)
    if args.dir:
        prefixes = tuple(p for p in args.prefix.split(",") if p)
        skip = set(p for p in args.skip.split(",") if p)
        files += [os.path.join(args.dir, fn) for fn in sorted(os.listdir(args.dir))
                  if fn.endswith(".md") and (not prefixes or fn.startswith(prefixes))
                  and fn not in skip]

    exit_code = 0
    for path in files:
        apath = os.path.abspath(path)
        root = args.root
        if not root:
            d = os.path.dirname(apath)
            while True:
                if os.path.exists(os.path.join(d, "AGENTS.md")) or os.path.exists(os.path.join(d, ".git")):
                    root = d
                    break
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        if not os.path.exists(apath) or not root:
            print("跳过（不存在或无库根）: %s" % path)
            exit_code = 2
            continue
        ok, n_link, n_ref, unresolved = convert_file(apath, Linker(root))
        tag = "改" if ok else "无改动"
        print("[%s %d 链 + %d 文献] %s" % (tag, n_link, n_ref, os.path.basename(path)))
        if unresolved:
            print("    [未解析 %d] %s" % (len(unresolved), "；".join(sorted(set(unresolved))[:8])))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
