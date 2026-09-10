# -*- coding: utf-8 -*-
r"""修复 验证脚本/注记-纠缠侧承载 中三处 \ne 撕裂。

成因同前：LaTeX 命令 \ne 的 \n 被当作转义符执行，断口处留下余部 "e"。
被撕裂的行以 "e 0$" 开头，需与前一行拼合，并在断口补回 \n 两字符。
"""
import io

P = '验证脚本/注记-纠缠侧承载：绕数 = 非定域关系通量的量子保护探路.md'

text = io.open(P, encoding='utf-8', newline='').read()
L = text.split('\n')
orig_n = len(L)

orphan = [i for i, s in enumerate(L) if s.startswith('e 0$')]
assert len(orphan) == 3, orphan
print('[ok] 定位到 3 处撕裂行:', [i + 1 for i in orphan])

# 自下而上处理，避免索引位移
for i in reversed(orphan):
    prev = L[i - 1].rstrip('\r')
    cur = L[i].rstrip('\r')
    assert prev.endswith('$[H, W] '), repr(prev[-30:])
    assert cur.startswith('e 0$'), repr(cur[:20])
    merged = prev + '\\n' + cur
    assert merged.count('$') % 2 == 0, merged.count('$')
    assert '\\ne 0$' in merged
    L[i - 1:i + 1] = [merged]
    print('    L%d+L%d 拼合 -> 该行 %d 字符' % (i, i + 1, len(merged)))

io.open(P, 'w', encoding='utf-8', newline='').write('\n'.join(L))
print('[ok] 已写回：%d 行 -> %d 行' % (orig_n, len(L)))
