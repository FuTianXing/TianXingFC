# -*- coding: utf-8 -*-
# 温度主式定案同步（批 A 第 3 部分）：核心洞见速览三节末尾新增第 26 条
import io

p = r'理论总纲\天行新学核心洞见速览.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

anchor = '再次神奇涌现。\n## 四、规范场和规范力'

new26 = ('26. **温度 = 场的复相干无序度，普朗克上限是定理**：$k_BT = E_P D$，'
         '$D = 1 - |\\langle z\\rangle|/\\sqrt{\\langle|z|^2\\rangle}$ 度量场强中未参与宏观相干的份额，'
         '柯西-施瓦茨定理给 $D \\in [0,1]$，普朗克温度 $T_P = E_P/k_B \\approx 1.4\\times10^{32}$ K 从量纲产物升级为定义上限，'
         '经典奇点发散被截断。温度独立于熵定义、覆盖相变前：原初混沌就是最热（$T = T_P$，元相海的属性），'
         '高温先于大爆炸存在，凝固与校准是降温史，今天 $D_0 = T_0/T_P \\approx 1.9\\times10^{-32}$（三十二个数量级）。'
         'CMB 涨落双分量无序度比 $2:1$ 构成可证伪检验；标准温度是本定义在低无序区的还原。'
         '（《温度的波粒互补定义：复相干无序度作为热力学标尺》，2026-09-08 定稿）\n')

assert s.count(anchor) == 1, 'anchor 匹配数 = %d' % s.count(anchor)
s = s.replace(anchor, '再次神奇涌现。\n' + new26 + '## 四、规范场和规范力', 1)

with io.open(p, 'w', encoding='utf-8', newline='') as f:
    f.write(s)
print('核心洞见速览 第 26 条插入完成')
