# -*- coding: utf-8 -*-
# README 登记元相常数词条 + 铁律检查
import io, re

p = r'wiki\README.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

old = '| [元相海](元相海.md) | 元相子的最大集合（波粒双侧、潜无穷、持续演化） | 物理概念 | 已确认 |'
new = old + '\n| [元相常数](元相常数.md) | 离散镶嵌落位概率的指数深度（$\\mathcal{D}_p = \\ln(1/p) \\approx 139$，由边缘几何与宇宙年龄导出） | 物理概念 | 数值已确认，命名纲领性 |'

assert s.count(old) == 1, 'old 匹配数 = %d' % s.count(old)
s = s.replace(old, new, 1)
with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('README 登记完成')

# 铁律检查新词条
p2 = r'wiki\元相常数.md'
with io.open(p2, 'r', encoding='utf-8') as f:
    s2 = f.read()
print('元相常数词条 | em-dash:', s2.count('\u2014'),
      '| 禁句式:', re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s2),
      '| 账隐喻:', re.findall(r'记账|账本|欠账', s2))
sections = re.findall(r'^## (.+)$', s2, re.M)
print('小节:', sections)
required = ['概述', '定义', '机制', '通俗类比', '例子', '与其他概念的关系', '参见', '参考资料']
print('八节齐全:', all(x in sections for x in required))
