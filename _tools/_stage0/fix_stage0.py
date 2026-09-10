# -*- coding: utf-8 -*-
r"""阶段 0 修复：P1-33 公式撕裂拼合 + P4-28 重复条目清理。

成因：LaTeX 命令 \ne 的 \n、\rangle 的 \r 被当作转义符执行，
断口处留下命令余部（\ne 剩 "e"，\rangle 剩 "angle"），
P1-33 这一条因此被拆成 6 个物理行。修复方式为按序拼合，
并在断口处补回被吃掉的 \n / \r 两字符。
"""
import io

P = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'

text = io.open(P, encoding='utf-8', newline='').read()
L = text.split('\n')
orig_n = len(L)

pieces = [L[i].rstrip('\r') for i in range(47, 53)]
expect_head = [
    '33. **有质量粒子运动机制',
    'e 0$ 全空间但基态空间内可对角化',
    'angle = (1-\\omega^{-2})W|gs',
    'angle ',
    'e 0$）与 $[H, W] ',
    'e 0$（激发部分）自洽',
]
expect_tail = [
    '影量子数（$[H, W] ',
    '（$[W, A_v]|gs',
    'ega^{-2})W|gs',
    'angle ',
    '0$）与 $[H, W] ',
    '第四至八节）',
]
for k in range(6):
    assert pieces[k].startswith(expect_head[k]), (k, pieces[k][:40])
    assert pieces[k].endswith(expect_tail[k]), (k, pieces[k][-40:])
print('[ok] 6 段撕裂残留断言通过')

joined = (pieces[0] + '\\n' + pieces[1]
          + '\\r' + pieces[2]
          + '\\r' + pieces[3]
          + '\\n' + pieces[4]
          + '\\n' + pieces[5])

for frag in ['（$[H, W] \\ne 0$ 全空间',
             '（$[W, A_v]|gs\\rangle = (1-\\omega^{-2})W|gs\\rangle \\ne 0$）与 $[H, W] \\ne 0$（激发部分）自洽']:
    assert frag in joined, frag
print('[ok] 拼合片段校验通过')

dup = L[132]
assert dup.startswith('28. **三常数数值被天行力精细调校的机制与允许区域**'), dup[:40]

L[47:53] = [joined + '\r']

idx = [i for i, s in enumerate(L) if s == dup]
assert len(idx) == 2, idx
del L[idx[1]]
print('[ok] P4-28 重复条目已删除，保留 index=%d' % idx[0])

io.open(P, 'w', encoding='utf-8', newline='').write('\n'.join(L))
print('[ok] 已写回：%d 行 -> %d 行；P1-33 拼合后 %d 字符' % (orig_n, len(L), len(joined)))
