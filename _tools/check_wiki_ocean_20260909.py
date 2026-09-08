# -*- coding: utf-8 -*-
"""铁律检查与格式核对：元相海词条 + AGENTS.md + README"""
import io, re

files = [r'wiki\元相海.md', r'AGENTS.md', r'wiki\README.md']
for p in files:
    with io.open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    emdash = s.count('\u2014\u2014') + s.count('\u2014')
    forbidden = re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s)
    ledger = re.findall(r'记账|账本|欠账', s)
    name = p.split('\\')[-1]
    print(name, '| em-dash:', emdash, '| 禁句式:', forbidden, '| 账隐喻:', ledger)

# 词条结构核对：八节齐全
with io.open(r'wiki\元相海.md', 'r', encoding='utf-8') as f:
    s = f.read()
sections = re.findall(r'^## (.+)$', s, re.M)
print('词条小节:', sections)
required = ['概述', '定义', '机制', '通俗类比', '例子', '与其他概念的关系', '参见', '参考资料']
missing = [x for x in required if x not in sections]
print('八节齐全:', not missing, ('缺失: %s' % missing) if missing else '')
