# -*- coding: utf-8 -*-
"""铁律检查：三处修订后的文件"""
import io, re

files = [r'时空的持续创生：元相子理论的膨胀机制.md',
         r'理论总纲\元相子理论的时空观：体系对时空的完整认识.md',
         r'内部草稿\注记-内生膨胀与边缘扩张的互补：本原创生的两种物侧显现.md']

for p in files:
    with io.open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    emdash = s.count('\u2014\u2014') + s.count('\u2014')
    forbidden = re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s)
    ledger = re.findall(r'记账|账本|欠账', s)
    name = p.split('\\')[-1][:24]
    print(name, '| em-dash:', emdash, '| 禁句式:', forbidden, '| 账隐喻:', ledger)
