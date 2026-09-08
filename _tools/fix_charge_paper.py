# -*- coding: utf-8 -*-
"""电荷论文三处小修：B1 遗留五项、B2 参考文献去编号、C1 投影写法。"""
import io, re

f = '论电荷的本质：从绕数到基态空间投影量子数的完整机制.md'
t = io.open(f, encoding='utf-8').read()

old1 = '遗留四项。其一，Z_N 推广的严格解析证明（当前为数值全系 + 解析骨架）。其二，先验纠缠到格点键关系自由度的严格映射（O1）。其三，有限增长晶格（环面拓扑替代）上的扇区结构。其四，规范-电荷同源的形式化（从先验纠缠构造 U(1) 冗余的端到端推导）。'
new1 = '遗留五项。其一，Z_N 推广的严格解析证明（当前为数值全系 + 解析骨架）。其二，先验纠缠到格点键关系自由度的严格映射（O1，已登记）。其三，有限增长晶格（环面拓扑替代）上的扇区结构。其四，规范-电荷同源的形式化（从先验纠缠构造 U(1) 冗余的端到端推导）。其五，跨扇区零矩阵元的严格解析证明（弦算符与局域生成元闭包的关系，§5.4 定理条件 3）。'
assert t.count(old1) == 1
t = t.replace(old1, new1)

s = t.find('## 参考文献')
head, tail = t[:s], t[s:]
tail = re.sub(r'^(\d+)\. ', '- ', tail, flags=re.M)
t = head + tail

old2 = '1. 投影本征方程：$P_{\\text{gs}} W |\\psi\\rangle = \\omega^m|\\psi\\rangle$'
new2 = '1. 投影本征方程：$(P_{\\text{gs}}WP_{\\text{gs}})|\\psi\\rangle = \\omega^m|\\psi\\rangle$'
assert t.count(old2) == 1
t = t.replace(old2, new2)

io.open(f, 'w', encoding='utf-8').write(t)
print('三处小修完成')
