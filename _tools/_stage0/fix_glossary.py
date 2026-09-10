# -*- coding: utf-8 -*-
r"""修复 理论总纲/术语表.md 第 145-146 行的表格撕裂。

成因同开放问题总览：LaTeX 命令 \nabla 的 \n 被当作转义符执行，
断口处留下命令余部 "abla"，"标架记录" 一行因此被拆成两个物理行，
符号列被截为 "$-" 与 "abla^2\Phi = s\,\delta$ |"。
修复方式为拼合两行，并在断口补回被吃掉的 \n 两字符。
"""
import io

P = '理论总纲/术语表.md'

text = io.open(P, encoding='utf-8', newline='').read()
L = text.split('\n')
orig_n = len(L)

a = L[144].rstrip('\r')
b = L[145].rstrip('\r')

assert a.startswith('| 标架记录 | 写入的稳态产物'), a[:40]
assert a.endswith('（拓扑性） | $-'), a[-30:]
assert b == r'abla^2\Phi = s\,\delta$ |', repr(b)
print('[ok] 断口断言通过')
print('    L145 尾: %r' % a[-24:])
print('    L146   : %r' % b)

joined = a + '\\n' + b

assert joined.count('|') == 4, joined.count('|')
assert joined.count('$') % 2 == 0, joined.count('$')
assert r'$-\nabla^2\Phi = s\,\delta$ |' in joined
print('[ok] 拼合校验通过：3 列，$ 配对，符号列 = %r' % r'$-\nabla^2\Phi = s\,\delta$')

L[144:146] = [joined]

io.open(P, 'w', encoding='utf-8', newline='').write('\n'.join(L))
print('[ok] 已写回：%d 行 -> %d 行' % (orig_n, len(L)))
print('    拼合后该行长度 %d 字符' % len(joined))
