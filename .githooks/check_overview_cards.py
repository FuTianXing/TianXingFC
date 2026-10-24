#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前检查：拒绝开放问题看板与档案页的形态回潮。

背景：2026-10-24 总览清理（18.3 万字符降致 4.7 万）暴露的根因是追加式登记——
每轮攻坚收尾向卡片追加一段带日期的流水，档案页出口存在却无人走。本钩子把
《AGENTS.md 开放问题同步规范》的四条形态纪律变成硬约束：

   一、单条目超 800 字符即拒绝（拆档红线，登记时当轮瘦身：早期动态移入档案页）；
   二、单行超 1200 字符即拒绝（巨行防线）；
   三、单条目内加粗日期块（**2026-MM-DD）达 3 个即拒绝——卡片里写历史段落；
   四、新增条目缺状态行给警告（不拒绝，允许例外，但每次提交都会提醒）。

检查对象为总览与已关闭清单两个看板文件，读暂存区版本。处置路径：
判据详情与历次登记一律先写入 内部草稿/开放问题档案/开放问题-<层级>-<编号>-<短题>.md
（无页即建），总览卡只做状态行替换与至多一行动态。

档案页防线（2026-10-24 增）：档案页是唯一被允许写历次判定的地方，但它同样不得
出现巨行与状态行续写。三条：

   五、档案页任一正文行超 600 字符即拒绝。档案页允许页大，不允许行大——十轮
      登记可以分十条，不得挤在一条里。实测存量最长 5407 字符、内嵌 6 个日期块。
   六、档案页状态行超 300 字符即拒绝。状态行只写当前状态，历次过程移到判据
      明细区。实测存量最长 4429 字符（P4-45）。
   七、档案页任一正文行内含 3 个以上日期块即拒绝，防止续写型巨行换个形态回来。

档案页的存量巨行不追溯（同 README 编号撞号的存量原则），本钩子只拦本次新增或
改动的行。已在本目录留 split_giant_lines.py 做一次性拆分。
"""
import os
import re
import subprocess
import sys

BOARDS = [
    # (路径, 是否执行 800 字符条目体量线)。总览是看板（800 拆档线适用）；
    # 已关闭清单是归档，条目按关闭流程须保留完整关闭依据与结论，不设体量线，
    # 只查单行巨行与历史段落回潮。
    ("内部草稿/开放问题总览：元相子理论待解决问题清单.md", True),
    ("内部草稿/开放问题已关闭清单：元相子理论已解决问题归档.md", False),
]
ARCHIVE_DIR = "内部草稿/开放问题档案"
LIMIT_CARD = 800
LIMIT_LINE = 1200
LIMIT_DATE_BLOCKS = 3
LIMIT_ARCHIVE_LINE = 600
LIMIT_ARCHIVE_STATUS = 300

ENTRY = re.compile(r"^(\d+)\. ")
CLOSED = re.compile(r"^### ((?:P\d+)-\S+)")
SEC = re.compile(r"^## (P\d)")
DATE_BLOCK = re.compile(r"\*\*20\d\d-\d\d-\d\d")
STATUS = re.compile(r"^\s+- 状态")


def staged_text(path):
    r = subprocess.run(["git", "show", ":" + path],
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def head_text(path):
    r = subprocess.run(["git", "show", "HEAD:" + path],
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def entries(text):
    """解析（节, 条目号, 起行, 止行, 是否状态行）。

    条目以「数字. 」开行，止于下一数字条目或任意二级标题；
    已关闭清单条目以「### P层-号」开行，止于下一 ### 或 --- 或文末。
    """
    out = []
    lines = text.split("\n")
    sec = None
    i = 0
    while i < len(lines):
        l = lines[i]
        m = SEC.match(l)
        if m:
            sec = m.group(1)
        elif l.startswith("## "):
            sec = None
        c = CLOSED.match(l)
        if c:
            code = c.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].startswith(("### ", "---")):
                j += 1
            out.append((code, i, j, lines[i:j]))
            i = j
            continue
        m = ENTRY.match(l)
        if m and sec:
            code = sec + "-" + m.group(1)
            j = i + 1
            while j < len(lines) and not re.match(r"^(\d+\. |## )", lines[j]):
                j += 1
            out.append((code, i, j, lines[i:j]))
            i = j
            continue
        i += 1
    return out


def check(path, card_limit, errors, warns):
    new = staged_text(path)
    if new is None:
        return
    old = head_text(path) or ""
    old_codes = {e[0] for e in entries(old)}
    for code, i, j, body in entries(new):
        n = sum(len(x) for x in body)
        if card_limit and n > LIMIT_CARD:
            errors.append("%s 条目 %s 体量 %d 字符，超 %d 上限。处置：把早期动态移入"
                          " 内部草稿/开放问题档案/ 的对应档案页（无页即建），卡片压回四行看板卡"
                          % (path, code, n, LIMIT_CARD))
        for k, ln in enumerate(body):
            if len(ln) > LIMIT_LINE:
                errors.append("%s 条目 %s 第 %d 行 %d 字符，超单行 %d 上限（巨行）"
                              % (path, code, i + k + 1, len(ln), LIMIT_LINE))
        k = len(DATE_BLOCK.findall("\n".join(body)))
        if k >= LIMIT_DATE_BLOCKS:
            errors.append("%s 条目 %s 含 %d 个加粗日期块（阈值 %d）——卡片里在写历史段落。"
                          "日期叙事只允许存在于档案页；状态行只写当前状态"
                          % (path, code, k, LIMIT_DATE_BLOCKS))
        if code not in old_codes and not any(STATUS.match(x) for x in body[1:]):
            warns.append("%s 新增条目 %s 缺状态行（首句之后的「   - 状态：…」行）"
                         % (path, code))


def head_bad_prefixes(text, limit):
    """HEAD 版中已超限行的前缀集合（取前 50 字符）。

    用途：存量不追溯。切分会让后续行号整体下移，`git diff` 把大批未改动的存量行
    标成新增，按行号判定「本次涉及」会把这些存量违规算到本轮头上（实测切分
    P1-55 时误报了 HEAD 中本就 637 字符的一行）。改用内容前缀判定：新行的前缀
    若命中某条 HEAD 已超限行，说明这是存量行的位移或轻改（如补句号），不算本轮
    新增违规。前缀取 50 字符，足以区分条目，又能容纳补句号一类的轻改。
    """
    out = set()
    for ln in text.split("\n"):
        if len(ln) > limit:
            out.add(ln[:50])
    return out


def added_prefixes(path):
    """暂存版相对 HEAD 真正新增或改写的行前缀集合（取前 50 字符）。

    只用 `+` 开头的行，不用 hunk 行号展开。切分会让后续行号整体下移，hunk 行号
    会把大批未改动的存量行算成「本次涉及」（实测切分 P1-55 时误报了 HEAD 中本就
    637 字符的一行）；改用实际新增行内容，位移行天然不在集合内。
    """
    r = subprocess.run(["git", "diff", "--unified=0", "HEAD", "--", path],
                       capture_output=True, text=True, encoding="utf-8")
    out = set()
    for ln in r.stdout.split("\n"):
        if ln.startswith("+") and not ln.startswith("+++"):
            out.add(ln[1:][:50])
    return out


def check_archives(errors):
    """档案页防线。只查本次新增或改写的行，存量巨行不追溯。"""
    if not os.path.isdir(ARCHIVE_DIR):
        return
    for fn in sorted(os.listdir(ARCHIVE_DIR)):
        if not fn.endswith(".md"):
            continue
        rel = ARCHIVE_DIR + "/" + fn
        new = staged_text(rel)
        if new is None:
            continue
        old = head_text(rel)
        if old == new:
            continue
        # 本轮真正动过的行（按内容前缀），以及 HEAD 中本就超限的前缀
        added = added_prefixes(rel)
        if not added:
            continue
        legacy = head_bad_prefixes(old, LIMIT_ARCHIVE_LINE)
        for i, ln in enumerate(new.split("\n")):
            if ln[:50] not in added:
                continue
            if ln[:50] in legacy:
                continue        # 存量行的位移或轻改（补句号等），不追溯
            n = len(ln)
            dates = len(DATE_BLOCK.findall(ln))
            if STATUS.match(ln) and n > LIMIT_ARCHIVE_STATUS:
                errors.append("%s 第 %d 行状态行 %d 字符，超 %d 上限。"
                              "状态行只写当前状态，历次过程移到判据明细区"
                              % (rel, i + 1, n, LIMIT_ARCHIVE_STATUS))
            elif n > LIMIT_ARCHIVE_LINE:
                errors.append("%s 第 %d 行 %d 字符，超档案页单行 %d 上限。"
                              "档案页允许页大、不允许行大：每轮登记单占一条"
                              % (rel, i + 1, n, LIMIT_ARCHIVE_LINE))
            elif dates >= 3:
                errors.append("%s 第 %d 行含 %d 个日期块（阈值 3）——续写型巨行。"
                              "每轮登记单占一条，不要往同一行追加"
                              % (rel, i + 1, dates))


def main():
    errors, warns = [], []
    for path, card_limit in BOARDS:
        check(path, card_limit, errors, warns)
    check_archives(errors)
    for w in warns:
        print("[警告] " + w, file=sys.stderr)
    if errors:
        print("[拒绝提交] 开放问题看板与档案页形态检查未过：", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
