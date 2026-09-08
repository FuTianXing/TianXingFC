# -*- coding: utf-8 -*-
# 第 24 条：指数深度补入准确定义公式
import io

p = r'理论总纲\天行新学核心洞见速览.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

old = '落位概率 $p = 3H_0 t_P \\approx 3.7\\times10^{-61}$（指数深度 139），由边缘几何与宇宙年龄导出。'
new = '落位概率 $p = 3H_0 t_P \\approx 3.7\\times10^{-61}$，指数深度 $\\mathcal{D}_p = \\ln(1/p) \\approx 139.15$（约 139），由边缘几何与宇宙年龄导出。'

assert s.count(old) == 1, 'old 匹配数 = %d' % s.count(old)
s = s.replace(old, new, 1)

with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('修正完成')
print('em-dash:', s.count('\u2014'))
