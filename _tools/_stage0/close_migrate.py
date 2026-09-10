# -*- coding: utf-8 -*-
r"""关闭项回迁：把 P1-32 从 P1 区移入「已关闭区」。

判据：
  - 原条目 状态：已关闭 且 当前判断 载明作者确认定案
  - 迁移保留原编号、关闭日期、结论与过程引用（改状态不删历史）
  - CRLF 行尾原样保留
"""
import io
import sys

P = r'D:\software\TianXingFC\内部草稿\开放问题总览：元相子理论待解决问题清单.md'

with io.open(P, 'r', encoding='utf-8', newline='') as f:
    raw = f.read()

assert '\r\n' in raw, 'CRLF expected'
L = raw.split('\r\n')

# 定位 P1-32 条目（1-indexed 53..59）
i = None
for k, s in enumerate(L):
    if s.startswith('32. **像素偏转与格点碰撞折返的冲突**'):
        i = k
        break
assert i is not None, 'P1-32 title not found'

block = L[i:i + 7]
assert block[0].startswith('32. **像素偏转与格点碰撞折返的冲突**'), block[0][:40]
assert block[1] == '', 'blank line after title expected'
assert block[2].startswith('    - 状态：已关闭'), block[2][:40]
assert block[3].startswith('    - 要点：'), block[3][:20]
assert block[4].startswith('    - 当前判断：'), block[4][:20]
assert block[5].startswith('    - 过程：'), block[5][:20]
assert block[6] == '', 'blank line after entry expected'
assert L[i + 7].startswith('33. **有质量粒子运动机制'), L[i + 7][:40]

title = block[0]
state = block[2][4:]      # 去掉 4 空格缩进
points = block[3][4:]
judge = block[4][4:]
proc = block[5][4:]

# 状态行改造：把「已关闭」扩写为带日期的关闭标记
new_state = state.replace('状态：已关闭 ｜', '状态：已关闭（2026-09-09 作者确认定案）｜', 1)
assert new_state != state, 'state rewrite failed'
# 当前判断 改标为 结论
new_judge = judge.replace('当前判断：', '结论：', 1)
assert new_judge != judge, 'judge relabel failed'
# 要点 改标为 要点（历史）
new_points = points.replace('要点：', '要点（历史）：', 1)
assert new_points != points, 'points relabel failed'

closed = [
    '### P1-32 像素偏转与格点碰撞折返的冲突',
    '',
    '关闭日期：2026-09-09 ｜ 关闭依据：作者确认定案，采用碰撞写入加标架联络方案，五层冲突全部消解闭环',
    '',
    new_state,
    new_points,
    new_judge,
    proc,
    '',
]

# 删除 P1 区原条目（含其后空行）
del L[i:i + 7]

# 定位「已关闭区」的（空）占位并替换
j = None
for k, s in enumerate(L):
    if s == '（空）':
        j = k
        break
assert j is not None, 'placeholder not found'
assert L[j - 1] == '', 'blank before placeholder expected'
L[j:j + 1] = closed

out = '\r\n'.join(L)
assert '\n' not in out.replace('\r\n', ''), 'stray LF introduced'
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(out)

print('OK: P1-32 migrated. lines %d -> %d, bytes %d -> %d' % (len(raw.split('\r\n')), len(L), len(raw), len(out)))
