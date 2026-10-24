#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前检查：拒绝开放问题看板的形态回潮。

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
"""
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
LIMIT_CARD = 800
LIMIT_LINE = 1200
LIMIT_DATE_BLOCKS = 3

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


def main():
    errors, warns = [], []
    for path, card_limit in BOARDS:
        check(path, card_limit, errors, warns)
    for w in warns:
        print("[警告] " + w, file=sys.stderr)
    if errors:
        print("[拒绝提交] 开放问题看板形态检查未过：", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
