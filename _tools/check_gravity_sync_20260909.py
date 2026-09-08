# -*- coding: utf-8 -*-
"""最终核对：像素偏转残留位置 + 铁律检查"""
import io, re, os

print('=== 理论总纲中残留"像素偏转"的位置 ===')
for root, dirs, files in os.walk(r'理论总纲'):
    for fn in files:
        if not fn.endswith('.md'):
            continue
        p = os.path.join(root, fn)
        with io.open(p, 'r', encoding='utf-8') as f:
            s = f.read()
        for i, ln in enumerate(s.split('\n'), 1):
            if '像素偏转' in ln:
                print('%s:%d: %s' % (fn, i, ln.strip()[:60]))

print()
print('=== 铁律检查（修改过的文件） ===')
files = [
    r'理论总纲\元相子理论要点总览.md',
    r'理论总纲\元相子理论已解决难题总览.md',
    r'理论总纲\元相子理论的时空观：体系对时空的完整认识.md',
    r'理论总纲\元相子理论综合综述：从心物波粒二象到量子力学的物侧基础.md',
    r'理论总纲\元相方程.md',
    r'理论总纲\螺旋阶梯几何.md',
    r'理论总纲\元相子理论的革命性：全面论证.md',
    r'理论总纲\术语表.md',
    r'理论总纲\子理论索引.md',
    r'内部草稿\开放问题总览：元相子理论待解决问题清单.md',
]
for p in files:
    with io.open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    emdash = s.count('\u2014')
    forbidden = re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s)
    ledger = re.findall(r'记账|账本|欠账', s)
    print('%s | em-dash: %d | 禁句式: %d | 账隐喻: %d' % (p.split('\\')[-1][:24], emdash, len(forbidden), len(ledger)))
