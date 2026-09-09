# -*- coding: utf-8 -*-
# 补修：第 3 节标题残留"同步相变"
import io

p = r'有质量粒子的形成：大爆炸同步相变的局域缺陷方案.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

old = '## 3 核心方案：基本粒子 = 同步相变的局域缺陷'
new = '## 3 核心方案：基本粒子 = 结晶扩张相变的局域缺陷'
assert s.count(old) == 1, 'old 匹配数 = %d' % s.count(old)
s = s.replace(old, new, 1)
with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('第 3 节标题修正完成')

# 复核残留
rem = [i for i, ln in enumerate(s.split('\n'), 1) if '同步相变' in ln]
print('残留行:', rem, '（应为标题 1、附录检查对象、复审记录历史描述）')
print('em-dash:', s.count('\u2014'))
