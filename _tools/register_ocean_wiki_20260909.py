# -*- coding: utf-8 -*-
"""AGENTS.md wiki 行明确为名词解释 + wiki/README.md 登记元相海词条"""
import io

def read(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()

def write(p, s):
    with io.open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)

# 1. AGENTS.md
p1 = r'AGENTS.md'
s1 = read(p1)
old1 = '| `wiki/` | 重要名词的通俗解释词条（wiki 风格：定义首句 + 元数据表 + 八节模板，见 `wiki/README.md`），每个名词一个文件 |'
new1 = '| `wiki/` | 重要名词的名词解释词条（wiki 风格：定义首句 + 元数据表 + 八节模板，见 `wiki/README.md`，内含现有词条索引），每个名词一个文件 |'
assert s1.count(old1) == 1, 'AGENTS.md old1 匹配数: %d' % s1.count(old1)
s1 = s1.replace(old1, new1, 1)
write(p1, s1)
print('AGENTS.md 更新成功')

# 2. wiki/README.md
p2 = r'wiki\README.md'
s2 = read(p2)
old2 = '| [拉格朗日量](拉格朗日量.md) | 作用量的被积函数，最小作用量原理 = 螺旋面极小曲面几何 | 物理概念 | 已确认 |'
new2 = old2 + '\n| [元相海](元相海.md) | 元相子的最大集合（波粒双侧、潜无穷、持续演化） | 物理概念 | 已确认 |'
assert s2.count(old2) == 1, 'README old2 匹配数: %d' % s2.count(old2)
s2 = s2.replace(old2, new2, 1)
write(p2, s2)
print('README 登记成功')
