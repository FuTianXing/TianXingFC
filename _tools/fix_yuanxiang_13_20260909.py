# -*- coding: utf-8 -*-
# 补修：元相方程第 13 行元相子定义中的旧表述
import io

p = r'理论总纲\元相方程.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

old = '几何刚度潜伏 $G$（时空像素偏转系数）'
new = '几何刚度潜伏 $G$（时空写入倾斜系数）'
assert s.count(old) == 1, 'old 匹配数 = %d' % s.count(old)
s = s.replace(old, new, 1)
with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('元相方程 第 13 行修正完成')
