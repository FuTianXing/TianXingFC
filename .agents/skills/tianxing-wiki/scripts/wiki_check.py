# -*- coding: utf-8 -*-
r'''
wiki_check.py <词条文件>...

按 wiki/README.md 的通用模板与写作要求检查词条。硬伤（退出码 1）：
缺一级标题；定义首句未加粗；元数据表缺字段；缺必需小节；控制字符。
提示（不影响退出码）：超长段落；ASCII 双引号；缺可选小节；残余书名号。
'''
import os
import re
import sys

REQUIRED_SECTIONS = ['定义', '机制', '参见', '参考资料']
OPTIONAL_SECTIONS = ['概述', '通俗类比', '例子', '与其他概念的关系']
META_FIELDS = ['分类', '定案状态', '首次出现', '相关词条']
DQ = chr(34)
TICK = chr(96)
FENCE = re.compile('^' + TICK * 3 + r'.*?^' + TICK * 3, re.S | re.M)
CODE = re.compile(TICK + r'[^' + TICK + r']*' + TICK)
WIKILINK = re.compile(r'\[\[[^\]]*\]\]')
EXEMPT_PREFIX = ('|', '#', '-', '>', '$')


def strip_noise(s):
    return WIKILINK.sub(' L ', CODE.sub(' C ', FENCE.sub(' ', s)))


def paragraphs(lines):
    blocks, cur = [], []
    for line in lines:
        t = line.strip()
        if not t:
            if cur:
                blocks.append('\n'.join(cur))
                cur = []
        elif t.startswith(EXEMPT_PREFIX) or re.match(r'^\d+\.', t):
            if cur:
                blocks.append('\n'.join(cur))
                cur = []
        else:
            cur.append(t)
    if cur:
        blocks.append('\n'.join(cur))
    return blocks


def check(path):
    hard, hint = [], []
    try:
        s = open(path, encoding='utf-8').read()
    except OSError as exc:
        return ['无法读取：%s' % exc], []
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', s):
        hard.append('控制字符')
    lines = s.split('\n')
    title = next((l for l in lines if l.startswith('# ')), None)
    if title is None:
        hard.append('缺一级标题')
    body = s.split(title, 1)[1] if title else s
    if not re.search(r'^\*\*[^*\n]+\*\*(?:[（(][^）)\n]{0,60}[）)])?\s*是', body, re.M):
        hard.append('缺定义首句（行首 **名词**是…）')
    for field in META_FIELDS:
        if not re.search(r'\|\s*' + field + r'\s*\|', s):
            hard.append('元数据表缺字段：%s' % field)
    secs = [m.group(1).strip() for m in re.finditer(r'^## (.+)$', s, re.M)]
    for sec in REQUIRED_SECTIONS:
        if not any(sec == x or sec in x for x in secs):
            hard.append('缺必需小节：%s' % sec)
    for sec in OPTIONAL_SECTIONS:
        if not any(sec == x or sec in x for x in secs):
            hint.append('缺可选小节：%s' % sec)
    for b in paragraphs(lines):
        if len(b) > 300:
            hint.append('超长段落 %d 字符：%s' % (len(b), b[:24]))
    if DQ in s:
        hint.append('含 ASCII 双引号 %d 处（体系惯例允许；落盘走 node_repl 时注意字符串边界）' % s.count(DQ))
    if re.search(r'《[^》]+》', s):
        hint.append('残余书名号 %d 处（工作区互引应为 wikilink）' % len(re.findall(r'《[^》]+》', s)))
    return hard, hint


def main(argv):
    files = [a for a in argv if not a.startswith('--')]
    if not files:
        print('用法：wiki_check.py <词条文件>...')
        return 2
    bad = 0
    for path in files:
        hard, hint = check(path)
        print('=' * 70)
        print(('!! ' if hard else 'ok ') + os.path.basename(path))
        for item in hard:
            print('   硬伤 | ' + item)
        for item in hint:
            print('   提示 | ' + item)
        if hard:
            bad += 1
    print('---- 共 %d 个词条，%d 个有硬伤 ----' % (len(files), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
