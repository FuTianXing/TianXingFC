# -*- coding: utf-8 -*-
# 最终核对：《有质量粒子的形成》复审修订结果
import io, re

p = r'有质量粒子的形成：大爆炸同步相变的局域缺陷方案.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

print('=== 残留"同步相变"位置（应为标题/历史记录/专名）===')
for i, ln in enumerate(s.split('\n'), 1):
    if '同步相变' in ln:
        print('%d: %s' % (i, ln.strip()[:70]))

print()
print('=== 铁律 ===')
print('em-dash:', s.count('\u2014'))
print('禁句式:', re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s))
print('账隐喻:', re.findall(r'记账|账本|欠账', s))

print()
print('=== 关键更新核对 ===')
checks = [
    ('结晶扩张相变', 8, '正文主表述已切换'),
    ('多核成案', 4, '多核成案标注'),
    ('复合位错粒子', 2, '粒子本体定案衔接'),
    ('基态空间投影量子数', 1, '电荷守恒机制更新'),
    ('周期狄拉克梳', 1, '元相方程 (II) 定案形式'),
    ('三物侧前提', 2, '自旋-统计三前提'),
    ('2026-09-09 复审', 1, '附录复审记录'),
]
for txt, expect, desc in checks:
    n = s.count(txt)
    print('%s: %d 处 (期望>=%d) %s' % (desc, n, expect, 'OK' if n >= expect else 'FAIL'))
