# -*- coding: utf-8 -*-
"""第 11 轮补：三处引用汇聚点的下游导航。

判据：只为「被多处承接的上游条目」补下游指针，使读者在上游能发现细化落点。
不追求图谱完全双向——承接链天然有向，机械对称化会制造冗余。
"""
import sys

P = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'
raw = open(P, 'rb').read().decode('utf-8')
orig = raw

EDITS = [
    # P4-20 是膨胀族枢纽（被 P4-21/P4-23/P4-30 引用），正文缺下游导航
    ('P4-20 补族内下游导航',
     r'开放问题，承接 P4-19）',
     r'开放问题，承接 P4-19）。膨胀史、暗能量与暴涨种子的对应条目见 P4-21、P4-23、P4-24。'),
    # P1-31 被 P1-4、P1-33、P4-27 三处承接，是缺陷密度攻坚落点
    ('P1-31 补下游承接导航',
     r'内部草稿；承接 P1-4、P1-11）',
     r'内部草稿；承接 P1-4、P1-11）。下游承接见 P1-33（运动环节）与 P4-27（成核接缝密度）。'),
    # P4-6 测量塌缩的三个角度
    ('P4-6 补第三角度指针',
     r'同题的另一角度见 P4-10 退相干独立几何推导）',
     r'同题的另一角度见 P4-10 退相干独立几何推导；相位采样角度见 P4-30 ④）'),
]

fail = 0
for desc, old, new in EDITS:
    n = raw.count(old)
    if n != 1:
        print(f'FAIL [{desc}] 匹配数={n}（应为 1）')
        fail += 1
        continue
    raw = raw.replace(old, new)
    print(f'OK   [{desc}]')

if fail:
    print(f'\n中止：{fail} 处未应用。')
    sys.exit(1)

ctrl = {c for c in raw if ord(c) < 32 and c not in '\r\n\t'}
if ctrl:
    print(f'中止：控制字符 {ctrl}')
    sys.exit(1)
if orig.count('\r\n') != raw.count('\r\n'):
    print('中止：CRLF 数变化')
    sys.exit(1)

with open(P, 'wb') as f:
    f.write(raw.encode('utf-8'))

print(f'\nOK: {len(EDITS)} 处应用')
print(f'字符 {len(orig)} -> {len(raw)}（{len(raw) - len(orig):+d}）')
