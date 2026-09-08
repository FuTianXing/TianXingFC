# -*- coding: utf-8 -*-
# 第 25 条格式修复：标题与正文数字前补空格
import io

p = r'理论总纲\天行新学核心洞见速览.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

edits = [
    ('25. 元相常数139：', '25. **元相常数 139**：'),
    ('139作为元相子理论中最神秘的数字组合', '139 作为元相子理论中最神秘的数字组合'),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, 'old 匹配数 = %d: %s' % (n, old[:30])
    s = s.replace(old, new, 1)
    print('替换成功:', old[:30])

with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('写入完成')
