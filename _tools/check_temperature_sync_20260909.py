# -*- coding: utf-8 -*-
# 最终核对：温度定案同步结果 + 铁律
import io, re

files = [
    (r'理论总纲\元相子理论要点总览.md', ['温度定义', '复相干无序度', 'k_BT = E_P D']),
    (r'理论总纲\元相子理论已解决难题总览.md', ['## 七、热力学与温度', '45. **温度定义', '46. **温度语义', '47. **温度宇宙学']),
    (r'理论总纲\天行新学核心洞见速览.md', ['26. **温度 = 场的复相干无序度']),
    (r'内部草稿\温度攻坚成果总览.md', ['v3 主式已作者裁决定案', '主式定案结案']),
]

for p, keys in files:
    with io.open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    for k in keys:
        print('%s | %s: %s' % (p.split('\\')[-1][:20], k, 'OK' if k in s else 'MISSING'))
    emdash = s.count('\u2014')
    forbidden = re.findall(r'不是[^。；]*而是|并非[^。；]*而是|不在于[^。；]*而在于', s)
    ledger = re.findall(r'记账|账本|欠账', s)
    print('  铁律 | em-dash: %d | 禁句式: %d | 账隐喻: %d' % (emdash, len(forbidden), len(ledger)))

# 残留"待作者最终裁决"
with io.open(r'内部草稿\温度攻坚成果总览.md', 'r', encoding='utf-8') as f:
    s = f.read()
print('温度总览残留"待作者最终裁决":', s.count('待作者最终裁决'))

# 已解决难题总览编号连续
with io.open(r'理论总纲\元相子理论已解决难题总览.md', 'r', encoding='utf-8') as f:
    s = f.read()
nums = re.findall(r'^(\d+)\. \*\*', s, re.M)
print('已解决难题编号:', nums)
