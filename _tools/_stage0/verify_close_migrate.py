# -*- coding: utf-8 -*-
r"""关闭项回迁后的全量校验。"""
import io
import re

P = r'D:\software\TianXingFC\内部草稿\开放问题总览：元相子理论待解决问题清单.md'
raw = io.open(P, 'rb').read()
txt = raw.decode('utf-8')
L = txt.split('\r\n')

print('== 行尾符 ==')
print('CRLF %d | lone LF %d' % (raw.count(b'\r\n'), raw.count(b'\n') - raw.count(b'\r\n')))

print('\n== 标题结构 ==')
for k, s in enumerate(L, 1):
    if s.startswith('#'):
        print('%4d  %s' % (k, s))

print('\n== 分层计数 ==')
cur = None
cnt = {}
order = []
for s in L:
    if s.startswith('## '):
        cur = s[3:].strip()
        if cur not in cnt:
            cnt[cur] = 0
            order.append(cur)
    elif cur and re.match(r'^\d+[a-z]?\. ', s):
        cnt[cur] += 1
for k in order:
    print('%-40s %d' % (k, cnt[k]))

print('\n== 状态字段分布 ==')
st = {}
for s in L:
    m = re.search(r'状态：([^\s｜]+)', s)
    if m:
        st[m.group(1)] = st.get(m.group(1), 0) + 1
for k, v in sorted(st.items(), key=lambda x: -x[1]):
    print('%-10s %d' % (k, v))

print('\n== 写作铁律 ==')
print('em-dash 计数：', txt.count('——') + txt.count('—'))
bad = [p for p in ('不是', '并非', '不在于') if p in txt]
pat = re.findall(r'不是[^，。；]{0,20}而是', txt) + re.findall(r'并非[^，。；]{0,20}而是', txt) + re.findall(r'不在于[^，。；]{0,20}而在于', txt)
print('禁句式命中：', pat if pat else '无')
print('"账"字计数：', txt.count('账'))

print('\n== 公式配对（逐行，跳过 $$ 块）==')
inblock = False
odd = []
for k, s in enumerate(L, 1):
    if s.strip().startswith('$$'):
        inblock = not inblock
        continue
    if inblock:
        continue
    if s.count('$') % 2:
        odd.append((k, s[:70]))
print('奇数 $ 行数：', len(odd))
for k, s in odd[:10]:
    print('  %4d %s' % (k, s))

print('\n== 已关闭区 ==')
i = next(k for k, s in enumerate(L) if s.startswith('## 已关闭区'))
for s in L[i:]:
    if s.startswith('---'):
        break
    if s.strip():
        print('  ' + s[:100])
