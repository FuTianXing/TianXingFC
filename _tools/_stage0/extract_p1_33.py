# -*- coding: utf-8 -*-
r"""外移 P1-33 过程记录。

P1-33「有质量粒子运动机制」原 18181 字符，为全文最长条目。前 151 字符是问题陈述，
其余 18030 字符是 2026-09-06 的约二十五条登记流水。载体经前缀匹配核对。
按作者指示：分级列表、回链加一句状态摘要、不搬公式。
"""
import io
import os
import re

P = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'

# 建全库文件名索引，用前缀匹配核对载体
names = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', 'images')]
    for f in files:
        if f.endswith('.md'):
            names.append(f[:-3])

text = io.open(P, encoding='utf-8', newline='').read()
L = text.split('\n')
orig_n, orig_chars = len(L), len('\n'.join(s.rstrip('\r') for s in L))

i = next(k for k, s in enumerate(L) if s.startswith('33. **有质量粒子运动机制'))
old = L[i].rstrip('\r')
assert len(old) == 18181, len(old)
print('[ok] 定位 P1-33，原文 %d 字符' % len(old))

cites = sorted({c for c in re.findall(r'《([^《》]{2,80})》', old)
                if re.match(r'(注记|总结报告|研究方向|评审|评估|温度攻坚|整改方案)', c)})
miss = [c for c in cites if not any(x == c or x.startswith(c) for x in names)]
print('[ok] 条目引用载体 %d 个，无法解析 %d 个' % (len(cites), len(miss)))
assert not miss, miss

block = [
    '33. **有质量粒子运动机制：σ 锁定 kink（涡旋环）传播候选**',
    '',
    '    - 状态：推进中 ｜ 层级：纲领 ｜ 来源：承接 P1-31 运动环节、P1-32 运动边界 ｜ 关联：P1-31、P1-32',
    '    - 要点：运动 = σ 锁定构型的边界传播（圆心不动、拓扑荷随行，与光传播同构），三维实现 = 三元组相位场上的最小尺度涡旋环。位形侧（粒性侧空间结构）的载体候选经三轮裁决实验全部否定：光滑场环在格点流下塌缩、单值场中孤立绕数载体不可构造、扇区极小强制离域。载体判定收敛到三元组格点内离散结构，与质量机制定案「缺陷 = 格点内异类」合流。',
    '    - 当前判断：绕数承载归波性侧（超选择层），质量归粒（格点内锁定），运动为两侧耦合的转换波，三块定案结构性归位。运动六项判据（锚定弱形式、惯性、测地响应线性、速度上限、能量动量不变量、协议独立）闭合，运动学验证升级为动力学闭环。键弹性原始形式经作者裁定为螺旋高度差（非紧致），无保留。遗留：三维复合位错形态与稳定性、锚核身份、落窗区宽度与色散同测、格点内离散保护的数值实现、凝固动力学形式化。',
    '    - 过程：见 `验证脚本/` 下 kink 三维稳定性、环的裁决、疤的裁决、扇区约束动力学、四件套复跑、相对论色散、弹性身份、测地响应、μ 差异、m_eff 定量、钉扎身份与钉扎重新评估等注记；`冲突诊断/注记-钉扎的重新评估`；根目录论文《绕数超选择：先验纠缠作为电荷保护层》《扇区约束动力学：两种绕数的裁决与场内局域载体不可能性》《从绕数困境到复合位错粒子：电荷载体问题的完整解决》《从复合位错到爱因斯坦场方程：粒子机制现状综合与攻坚纲领》；`内部草稿/` 下《注记-复合模型》《注记-新模型下的叠加态与概率幅》《粒子机制阶段总结与场方程攻坚立项》。',
    '',
]
new_chars = sum(len(x) for x in block)

L[i:i + 1] = [x + '\r' for x in block]

out = '\n'.join(L)
io.open(P, 'w', encoding='utf-8', newline='').write(out)

print('[ok] P1-33：%d -> %d 字符，压缩 %.0f%%' % (len(old), new_chars, 100 * (1 - new_chars / len(old))))
print('[ok] 行数 %d -> %d；全文字符 %d -> %d' % (orig_n, len(L), orig_chars, len('\n'.join(s.rstrip('\r') for s in L))))
