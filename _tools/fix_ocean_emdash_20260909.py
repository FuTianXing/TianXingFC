# -*- coding: utf-8 -*-
"""修正元相海词条中文档标题的 em-dash：改用体系惯例简称"""
import io

p = r'wiki\元相海.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

old1 = '| 首次出现 | 补充注释 24（《元相子几何动力学——束缚振荡修订整合版》，2026-08-16） |'
new1 = '| 首次出现 | 补充注释 24（主文档《元相子几何动力学》，2026-08-16） |'
assert s.count(old1) == 1
s = s.replace(old1, new1, 1)

old2 = '- 补充注释 24：元相海（《元相子几何动力学——束缚振荡修订整合版》，2026-08-16）'
new2 = '- 补充注释 24：元相海（主文档《元相子几何动力学》，2026-08-16）'
assert s.count(old2) == 1
s = s.replace(old2, new2, 1)

with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('修正完成，剩余 em-dash:', s.count('\u2014'))
