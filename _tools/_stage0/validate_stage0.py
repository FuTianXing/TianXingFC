# -*- coding: utf-8 -*-
r"""阶段 0 校验：结构完整性、公式配对、禁用字符、改动范围核对。"""
import io, re

P = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'
B = '_tools/_stage0/开放问题总览.bak-20260911.md'

raw = io.open(P, 'rb').read()
print('== 1. 编码与行尾符 ==')
print('UTF-8 BOM:', raw[:3] == b'\xef\xbb\xbf')
print('CRLF:', raw.count(b'\r\n'), '| bare LF:', raw.count(b'\n') - raw.count(b'\r\n'),
      '| bare CR:', raw.count(b'\r') - raw.count(b'\r\n'))

new = io.open(P, encoding='utf-8', newline='').read().split('\n')
bak = io.open(B, encoding='utf-8', newline='').read().split('\n')
newc = [s.rstrip('\r') for s in new]
bakc = [s.rstrip('\r') for s in bak]

print()
print('== 2. 改动范围核对（应仅两处）==')
joined = newc[47]
expect = bakc[:47] + [joined] + bakc[53:132] + bakc[133:]
print('除 P1-33 拼合与 P4-28 去重外，其余行逐字一致:', newc == expect)
diff = [i for i in range(min(len(newc), len(bakc))) if newc[i] != bakc[i]]
print('首个差异行 index:', diff[0] if diff else None, '（预期 47）')

print()
print('== 3. 公式配对 ==')
bad_dollar = [i + 1 for i, s in enumerate(newc) if s.count('$') % 2]
print('$ 数为奇数的行:', bad_dollar if bad_dollar else '无')
bad_brace = [i + 1 for i, s in enumerate(newc) if s.count('{') != s.count('}')]
print('{} 不配对的行:', bad_brace if bad_brace else '无')
bad_paren = [i + 1 for i, s in enumerate(newc) if s.count('（') != s.count('）')]
print('全角括号不配对的行:', bad_paren if bad_paren else '无')

print()
print('== 4. 撕裂残留复检 ==')
pat = re.compile(r'^(e 0\$|angle|rangle|abla|eq )')
orphan = [(i + 1, s[:60]) for i, s in enumerate(newc) if pat.match(s)]
print('残留行:', orphan if orphan else '无')

print()
print('== 5. 禁用字符扫描 ==')
t = '\n'.join(newc)
print('中文 em-dash:', t.count('——') + t.count('—'))
print('"不是...而是"及变体:',
      len(re.findall(r'不是.{0,20}而是', t)) + len(re.findall(r'并非.{0,20}而是', t)))
print('"账"字:', t.count('账'))

print()
print('== 6. 体量与条目数 ==')
print('行数:', len(newc), '| 字符数:', len(t))
print('条目数:', len(re.findall(r'(?m)^\d+[a-z]?\.\s', t)))
big = sorted(((len(s), i + 1) for i, s in enumerate(newc)), reverse=True)[:5]
print('最长 5 行（字符数, 行号）:', big)
