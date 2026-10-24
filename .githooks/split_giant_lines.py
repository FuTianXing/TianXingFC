#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_archive_giant_lines.py — 把档案页的巨行按轮次拆成多条。

背景：档案页是唯一被允许写历次判定的地方，但「页可以大」被误读成「行可以长」。
存量 12 行超 600 字符，最长 2525 字符（P1-62 第 157 行，内嵌 6 个轮次登记）。

切分规则（保守，只切有把握的）：

  规则一，轮次段切分。一行内含多个「日期段」起首的记录时，从每个日期段起点
    断开，前一条末尾补句号。日期段的起点判定用后顾边界，避免把文件名里的
    日期（如 `评审报告-xxx-2026-09-11.md`）误当成分段点：前一字符须是
    。；：）」] 五种之一。这是续写型巨行的主因。

  规则二，状态行切分。状态行超长时，把「（YYYY-MM-DD …」起的历次过程段拆为
    独立的「- 过程」行，状态行只留当前状态。

不动的内容（三类，脚本不猜）：
  - markdown 表格行（P1-58 那种整表都是长行的）。表格按列对齐，拆行即毁表，
    须人工改成一轮一列表格或改成列表。
  - 单轮写太长的列表项。没有轮次边界可切，切了只是把一句话截断，须人工改写。
  - 表格行与单轮长行由本脚本报告清单，交人工处理。
"""
import os
import re
import sys

ARCHIVE_DIR = "内部草稿/开放问题档案"
LIMIT = 600
# 轮次段起点：前一字符是 。；：）」] 之一，后跟可选的 ** 与日期
SEG_START = re.compile(r"(?<=[。；：）」\]])(\*\*)?(20\d\d-\d\d-\d\d)")
BULLET = re.compile(r"^(\s*)- (.*)$", re.S)


def split_by_rounds(line):
    """按轮次段起首切分。返回 None 表示无法安全切分。"""
    body = line.rstrip("\n")
    m = BULLET.match(body)
    if not m:
        return None
    indent, text = m.group(1), m.group(2)
    starts = [mm.start() for mm in SEG_START.finditer(text)]
    if len(starts) < 2:
        return None
    parts = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        parts.append(text[s:e].rstrip())
    head = text[:starts[0]].rstrip()
    if head:
        parts.insert(0, head)
    out = []
    for p in parts:
        p = p.rstrip()
        if not p:
            continue
        if not p.endswith(("。", "；", "：", "）", ")", "」", "]")):
            p = p + "。"
        out.append(indent + "- " + p)
    return out


def split_status(line):
    """状态行超长时，把（日期 …）起的过程段拆到独立行。"""
    body = line.rstrip("\n")
    m = BULLET.match(body)
    if not m or not body.lstrip().startswith(("- 状态", "-状态")):
        return None
    indent = m.group(1)
    rest = m.group(2)
    # 状态行形如「状态：开放（2026-10-17 …；2026-10-18 …）」
    cm = re.match(r"^状态[:：]\s*(.*)$", rest)
    if not cm:
        return None
    inner = cm.group(1)
    # 找第一个「（YYYY-MM-DD」
    pm = re.search(r"[（(]20\d\d-\d\d-\d\d", inner)
    if not pm:
        return None
    head = inner[:pm.start()].rstrip()
    tail = inner[pm.start():]
    if len(head) > LIMIT:
        return None
    out = [indent + "- 状态：" + head]
    # tail 按「；」与日期块切开成多条过程行
    pieces = re.split(r"(?=\*\*?20\d\d-\d\d-\d\d)", tail)
    buf = ""
    for p in pieces:
        if not p.strip():
            continue
        if buf and len(buf) + len(p) > LIMIT:
            out.append(indent + "  - 过程：" + buf.strip().rstrip("；"))
            buf = p
        else:
            buf += p
    if buf.strip():
        out.append(indent + "  - 过程：" + buf.strip().rstrip("；"))
    return out


def main():
    apply = "--apply" in sys.argv
    split_ok, manual = [], []
    for fn in sorted(os.listdir(ARCHIVE_DIR)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(ARCHIVE_DIR, fn)
        raw = open(path, encoding="utf-8", newline="").read()
        lines = raw.split("\n")
        out, changed = [], 0
        for ln in lines:
            if len(ln) <= LIMIT:
                out.append(ln)
                continue
            res = split_by_rounds(ln) or split_status(ln)
            if res and all(len(x) <= LIMIT for x in res) and len(res) > 1:
                out.extend(res)
                changed += 1
                split_ok.append((fn, len(ln), len(res), res))
            else:
                out.append(ln)
                # 分类不可切的原因，交人工
                if ln.lstrip().startswith("|"):
                    why = "表格行（拆行毁表）"
                elif split_by_rounds(ln) or split_status(ln):
                    why = "切后仍有超限行"
                else:
                    why = "单轮长项，无轮次边界"
                manual.append((fn, len(ln), why))
        if changed and apply:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(out))

    print("=== 已切分 ===")
    for fn, before, n, res in split_ok:
        print("%s：%d 字符 → %d 条" % (fn, before, n))
        for r in res:
            print("    " + r[:76])
    print()
    print("=== 须人工处理 ===")
    for fn, n, why in manual:
        print("%s：%d 字符，%s" % (fn, n, why))
    print()
    print("可切分 %d 行，须人工 %d 行" % (len(split_ok), len(manual)))
    if not apply:
        print("[dry] 未落盘。加 --apply 执行。")
    else:
        print("[apply] 已写 %d 个文件" % len({r[0] for r in split_ok}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
