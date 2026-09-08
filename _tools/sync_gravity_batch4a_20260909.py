# -*- coding: utf-8 -*-
"""碰撞写入方案定案同步（批 4a）：子理论索引状态确认化"""
import io

def read(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()

def write(p, s):
    with io.open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)

p = r'理论总纲\子理论索引.md'
s = read(p)

edits = [
    ('- **碰撞写入标架几何：引力场的碰撞写入与费米子的转换波运动** 🔄（2026-09-06 正稿新成，未评审）',
     '- **碰撞写入标架几何：引力场的碰撞写入与费米子的转换波运动** ✅（2026-09-06 正稿，2026-09-09 作者确认定案）'),
    ('- **碰撞写入标架几何：引力场的碰撞写入与费米子的转换波运动** 📝 修订中（2026-09-06 新成稿，未评审）',
     '- **碰撞写入标架几何：引力场的碰撞写入与费米子的转换波运动** ✅ 已定稿（2026-09-06 成稿，2026-09-09 作者确认）'),
]
for old, new in edits:
    n = s.count(old)
    assert n == 1, '子理论索引 old 匹配数 = %d: %s' % (n, old[:40])
    s = s.replace(old, new, 1)
    print('子理论索引 替换成功')

write(p, s)
print('子理论索引 写入完成')
