# -*- coding: utf-8 -*-
r"""外移 P1-5 过程记录；并为试点条目 P4-29 补空行。

P1-5「爱因斯坦场方程的完整导出」原 2854 字符，几乎全部是 2026-09-10 至 09-11 的
登记流水。载体十项经核对全部存在，外移为删除摘要、保留回链。
按作者指示：分级列表、回链加一句状态摘要、不搬公式。
"""
import io

P = '内部草稿/开放问题总览：元相子理论待解决问题清单.md'

text = io.open(P, encoding='utf-8', newline='').read()
L = text.split('\n')
orig_n, orig_chars = len(L), len('\n'.join(s.rstrip('\r') for s in L))

# --- 1. 试点条目 P4-29 补空行，保证嵌套列表渲染稳定 ---
j = next(k for k, s in enumerate(L) if s.startswith('29. **洛伦兹涌现链的形式化遗留**'))
if L[j + 1].strip() != '':
    L[j + 1:j + 1] = ['\r']
    print('[ok] P4-29 标题后补空行')

# --- 2. 外移 P1-5 ---
i = next(k for k, s in enumerate(L) if s.startswith('5. 爱因斯坦场方程的完整导出'))
old = L[i].rstrip('\r')
assert len(old) == 2854, len(old)
assert '2026-09-10 张量桥分段 (a) 完成登记' in old
assert 'P1-5 剩余' in old
print('[ok] 定位 P1-5，原文 %d 字符' % len(old))

block = [
    '5. **爱因斯坦场方程的完整导出**（当前仅静态弱场）',
    '',
    '    - 状态：推进中 ｜ 层级：纲领 ｜ 来源：总纲 §13、《割裂的终结》开放问题 ｜ 关联：P1-32、P1-34、P1-34a',
    '    - 要点：静态弱场已由标量到张量定量桥分段 (a)(b) 交付。仍待攻为度规识别的微观导出、强场非线性、分段 (d) 源项微观分解（门控 C1）、动态波方程的完整协变形式。另有一项记号约定待提升为工作区统一记号（带符号牛顿势与幅值记号的换算，关联 P1-34）。',
    '    - 当前判断：主战役四分段中 (a)(b) 完成，(c) 潮汐项与里奇对应与 (d) 待 C1 后推进。《爱因斯坦场方程的微观根基》已按碰撞写入重写完成，子理论索引状态 ⚠️ 转 ✅，待作者确认转正。',
    '    - 过程：见《注记-张量桥分段a攻坚计划》《注记-张量桥分段a S1探测结果》《注记-张量桥分段b攻坚计划》，`验证脚本/` 下分段 a、分段 b、法线场动力学健康性三份注记与对应脚本；文档整改见《整改方案-引力文档状态与措辞收紧》；评审与逐条修复见 `评审报告/评审报告-碰撞写入与标架联络-2026-09-11.md` 与同名修改报告。',
    '',
]
new_chars = sum(len(x) for x in block)

L[i:i + 1] = [x + '\r' for x in block]

out = '\n'.join(L)
io.open(P, 'w', encoding='utf-8', newline='').write(out)

print('[ok] P1-5：%d -> %d 字符，压缩 %.0f%%' % (len(old), new_chars, 100 * (1 - new_chars / len(old))))
print('[ok] 行数 %d -> %d；全文字符 %d -> %d' % (orig_n, len(L), orig_chars, len('\n'.join(s.rstrip('\r') for s in L))))
