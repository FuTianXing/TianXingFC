# -*- coding: utf-8 -*-
# README 条目行格式守卫：只校验本次提交在 验证脚本/README.md 中新增或改写的条目行，存量不追溯。
# 目的：防止读数、判语、边界、过程记录等正本内容回流索引（2026-10-24 冗余治理的常驻防线）。
# 条目行两种合法形态（行长一律 ≤300 字符）：
#   占位行：- **N.** 占位：……（≤150 字符，只写占位标记）
#   有注记：- **N.** [脚本名：][描述 ≤30 字：][[链接]]（日期），判据计数与状态标记白名单尾段。
#           链接指向历史条目分档时日期可省。
#   无注记：- **N.** `脚本名`：短语 ≤60 字（日期）。   （无独立注记的轮次专用）
import re
import subprocess
import sys

README = '验证脚本/README.md'
TAIL_SEG = re.compile(
    r'^(判据 .{1,50}'
    r'|[A-Za-z][\w.\-]{0,10} 判(否|负)'
    r'|读数作废|已归档'
    r'|第 \d+[a-z]? 号撤回'
    r'|经第 \d+ 号更正)$')
SCRIPTS = re.compile(r'^`[\w.+-]+\.py`( \+ `[\w.+-]+\.py`)*$')


def check_entry(line):
    """校验单行，返回 None 或错误说明。"""
    if len(line) > 300:
        return '行长 %d 超 300：读数、判语、边界、过程记录等正本写验证注记，索引行只留五要素' % len(line)
    m = re.match(r'^- \*\*(\d+[a-z]?)\.\*\* (.+)$', line)
    if not m:
        return '缺「- **编号.** 」骨架'
    rest = m.group(2)
    if rest.startswith('占位'):
        if len(line) > 150:
            return '占位行超 150 字符：占位行只写占位标记，不写判据与结论'
        return None
    mb = re.match(r'^`[\w.+-]+\.py`：([^[]{1,60})（(\d{4}-\d{2}-\d{2})）。$', rest)
    if mb:
        return None
    li = re.search(r'\[\[([^\]]+)\]\]', rest)
    if not li:
        return '无 [[注记]] 链接：有注记轮次用 [[链接]]，无注记轮次用「`脚本`：短语（日期）。」形态'
    if rest.count('[[') != 1:
        return '多于一个链接：通俗版、更正依据等第二链接在注记内给，索引只留注记一条'
    head = rest[:li.start()]
    if head:
        if not head.endswith('：'):
            return '链接前的前缀须以全角冒号收尾（脚本名或 ≤30 字描述）'
        parts = head[:-1].split('：')
        if len(parts) == 1:
            if not (SCRIPTS.match(parts[0]) or len(parts[0]) <= 30):
                return '链接前前缀不合规：%s' % parts[0]
        elif len(parts) == 2:
            if len(parts[0]) > 30:
                return '链接前描述前缀超 30 字：%s' % parts[0]
            if not SCRIPTS.match(parts[1].strip()):
                return '脚本名段不合规：%s' % parts[1]
        else:
            return '链接前前缀段数超限：%s' % head
    target = li.group(1)
    after = rest[li.end():]
    dm = re.match(r'（(\d{4}-\d{2}-\d{2})）', after)
    if not dm:
        if '历史条目' in target:
            after_tail = after
            if after_tail not in ('', '。'):
                return '历史条目行链接后只允许句号，现为：%s' % after_tail[:40]
            return None
        return '链接后须紧跟（YYYY-MM-DD 日期）'
    tail = after[dm.end():]
    if tail not in ('', '。'):
        if not tail.endswith('。'):
            return '行须以句号收尾'
        for seg in re.split(r'[，；]', tail[:-1]):
            if seg and not TAIL_SEG.match(seg):
                return '尾段「%s」不在状态词白名单（判否/判负/读数作废/第 N 号撤回/已归档/经第 N 号更正/判据计数）；数值、判语、边界等写注记' % seg
    return None


def main():
    out = subprocess.run(
        ['git', 'diff', '--cached', '-U0', '--', README],
        capture_output=True).stdout.decode('utf-8', 'ignore')
    errors = []
    for hunk in out.split('\n'):
        if not hunk.startswith('+') or hunk.startswith('+++'):
            continue
        line = hunk[1:]
        if not re.match(r'^- \*\*\d', line):
            continue
        msg = check_entry(line.rstrip('\r'))
        if msg:
            errors.append('  %s' % msg)
    if errors:
        print('README 条目行格式检查未通过（规则见 AGENTS.md 数值验证脚本规范）：')
        for e in errors:
            print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
