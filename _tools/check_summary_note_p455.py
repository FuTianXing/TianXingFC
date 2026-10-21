#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4-55 总结注记写作铁律自检。

用法：python _tools/check_summary_note_p455.py <文件路径>
"""
import io
import re
import sys

MAX_PARA = 300

# 检查项
EM_DASH = re.compile(r"[\u2014\u2015]")
NOT_BUT = re.compile(r"不是[^。；\n]{0,40}?而是")
ERFEI = re.compile(r"而非|并非")
LEDGER = re.compile(r"记账|账本|欠账")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WIKI_ALIAS = re.compile(r"\[\[[^\]\n]*?(?<!\\)\|")

# 代码跨度内豁免（反引号包裹的行内片段、以及整行代码块）
FENCE = re.compile(r"^\s*```")


def strip_code_spans(text):
    """把反引号包裹的片段与代码块整段置空，避免误报。"""
    out = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        if in_fence:
            out.append("")
            continue
        line = re.sub(r"`[^`\n]*`", "", line)
        out.append(line)
    return "\n".join(out)


def is_table_line(line):
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 1


def split_cells(row):
    """按未转义的竖线切分表格行，返回单元格列表。"""
    cells, buf, i = [], [], 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row) and row[i + 1] == "|":
            buf.append("\\|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    cells.append("".join(buf))
    return cells


def main(path):
    raw = io.open(path, encoding="utf-8", newline="").read()
    problems = []

    # 换行符与编码
    if "\r\n" in raw:
        problems.append("[CRLF] 文件含 CRLF 换行符")
    if raw and not raw.endswith("\n"):
        problems.append("[EOF] 文件末尾缺换行")

    text = strip_code_spans(raw)
    lines = text.split("\n")

    # 一、em-dash
    for i, line in enumerate(lines, 1):
        if EM_DASH.search(line):
            problems.append("[em-dash] L%d: %s" % (i, line.strip()[:90]))

    # 二、不是……而是
    for i, line in enumerate(lines, 1):
        m = NOT_BUT.search(line)
        if m:
            problems.append("[不是..而是] L%d: %s" % (i, line.strip()[:90]))

    # 三、而非 / 并非
    for i, line in enumerate(lines, 1):
        m = ERFEI.search(line)
        if m:
            problems.append("[而非/并非] L%d: %s" % (i, line.strip()[:90]))

    # 四、记账隐喻
    for i, line in enumerate(lines, 1):
        m = LEDGER.search(line)
        if m:
            problems.append("[记账隐喻] L%d: %s" % (i, line.strip()[:90]))

    # 七、控制字符
    for i, line in enumerate(lines, 1):
        m = CTRL.search(line)
        if m:
            problems.append("[控制字符] L%d: %r" % (i, m.group(0)))

    # 六、表格裸竖线：逐表比对行内未转义竖线的切分数
    header_cols = None
    for i, line in enumerate(lines, 1):
        if not is_table_line(line):
            header_cols = None
            continue
        cells = split_cells(line.strip())
        # 首尾为分隔符，切分结果首尾各有一个空串
        body = cells[1:-1]
        if set("".join(body).replace("|", "").replace("-", "").strip()) <= set():
            continue  # 分隔行
        if header_cols is None:
            header_cols = len(body)
            continue
        if len(body) != header_cols:
            problems.append("[表格裸竖线] L%d: 单元格数 %d，表头 %d" % (i, len(body), header_cols))

    # wikilink 别名未转义（规则只约束表格单元格内的别名竖线）
    for i, line in enumerate(lines, 1):
        if not is_table_line(line):
            continue
        if WIKI_ALIAS.search(line):
            problems.append("[wikilink别名未转义] L%d: %s" % (i, line.strip()[:90]))

    # 五、段落超长：按空行分块；标题、表格行、列表项、水平线豁免
    blocks, cur = [], []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    for blk in blocks:
        first = blk[0].strip()
        if first.startswith("#"):
            continue
        if all(is_table_line(x) for x in blk):
            continue
        if all(re.match(r"^\s*([-*+]|\d+\.)\s", x) for x in blk):
            continue  # 列表项整块豁免
        if all(re.match(r"^\s*([-*_=]\s*){3,}$", x) for x in blk):
            continue
        body = "".join(x.strip() for x in blk)
        if len(body) > MAX_PARA:
            problems.append("[段落超长] 长度 %d > %d，起首: %s" % (len(body), MAX_PARA, first[:50]))

    # 单个列表项也提示（不计入失败，仅列出超长项）
    long_items = []
    for blk in blocks:
        if all(re.match(r"^\s*([-*+]|\d+\.)\s", x) for x in blk):
            for x in blk:
                s = x.strip()
                if len(s) > MAX_PARA:
                    long_items.append(len(s))

    print("检查文件：%s" % path)
    print("字节数：%d，行数：%d" % (len(raw.encode("utf-8")), raw.count("\n")))
    if long_items:
        print("提示：列表项超 %d 字符的有 %d 条（AGENTS.md 允许列表项例外）" % (MAX_PARA, len(long_items)))
    if problems:
        print("发现 %d 处问题：" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("全部检查项通过。")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python %s <文件路径>" % sys.argv[0])
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
