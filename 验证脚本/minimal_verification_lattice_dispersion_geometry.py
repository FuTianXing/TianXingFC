# -*- coding: utf-8 -*-
r"""
minimal_verification_lattice_dispersion_geometry.py

[洛伦兹线说明轮] 首阶偏离的几何来源与方向指纹（判据 D1 至 D5）

对应论文：《元相子振荡晶格的色散关系：低能光速涌现与洛伦兹破坏首阶项》。
本轮不改结论，只把"为什么偏离、公式的几何意义、与晶格哪个特征对应"三问钉在可复现读数上。
全部计算在无量纲量 q_a = k_a l_P 上进行（单位口径统一，避免 k 与 k l_P 混用）：

    omega^2 = (4/t_P^2) sum_a sin^2(q_a/2)，展开为 (1/t_P^2)[Q^2 - (1/12) sum_a q_a^4 + O(q^6)]，Q^2 = sum q_a^2

判据
D1 精确色散对展开：四阶系数 1/12，余项按 q^6（log-log 斜率 6）
D2 方向指纹：sum_a n_a^4 = 1、1/2、1/3（6:3:2）；球面平均 3/5
D3 次光速界：|v_g|^2 / c^2 = sum sin^2(q_a) / (4 sum sin^2(q_a/2)) <= 1
D4 两约定等价：1 eV 与 1 TeV 的 delta omega/omega 逐位相同
D5 登记

运行：python 验证脚本/minimal_verification_lattice_dispersion_geometry.py（确定性，约 2 秒）
"""
import numpy as np
import math
import time

T_START = time.time()
lP = 1.616255e-35
c = 2.99792458e8
hbar = 1.054571817e-34
eV = 1.602176634e-19
tP = lP / c
EP_t = hbar / tP          # 按拍长定义，焦耳
EP_t_eV = EP_t / eV

def S2(qvec):
    """sum_a sin^2(q_a/2)（无量纲）"""
    q = np.asarray(qvec, dtype=float)
    return float(np.sum(np.sin(q / 2.0) ** 2))

def S1(qvec):
    """sum_a sin^2(q_a)（无量纲）"""
    q = np.asarray(qvec, dtype=float)
    return float(np.sum(np.sin(q) ** 2))

print("=" * 78)
print("首阶洛伦兹破坏的几何来源与方向指纹（判据 D1 至 D5）")
print("=" * 78)

print()
print("[D1] 精确色散对展开（沿 [100]，全部用 q = k l_P）")
for n in (2, 3, 4, 5, 6):
    q = 10.0 ** (-n)
    ex = 4.0 * S2([q, 0, 0])
    ap = q ** 2 - q ** 4 / 12.0
    print("     q = 1e-%d：精确 %.14e ｜ 展开 %.14e ｜ 相对差 %.2e" % (n, ex, ap, abs(ex - ap) / ex))
qs_id = np.linspace(1e-3, 3.0, 4001)
iddev = np.max(np.abs(4.0 * np.sin(qs_id / 2.0) ** 2 - 2.0 * (1.0 - np.cos(qs_id))))
print("     恒等核对：max |4 sin^2(q/2) - 2(1-cos q)| = %.2e（期望机器零，两写法同源）" % iddev)
coef = []
for n in (3, 4, 5, 6):
    q = 10.0 ** (-n)
    coef.append((q ** 2 - 4.0 * S2([q, 0, 0])) / q ** 4)
print("     四阶系数（(Q^2 - 精确)/sum q^4）：%s（期望 1/12 = %.10f）"
      % (" ".join("%.12f" % v for v in coef), 1.0 / 12.0))
slopes = []
qs = [0.4, 0.2, 0.1, 0.05]
for q1, q2 in zip(qs[:-1], qs[1:]):
    e1 = abs(4.0 * S2([q1, 0, 0]) - (q1 ** 2 - q1 ** 4 / 12.0))
    e2 = abs(4.0 * S2([q2, 0, 0]) - (q2 ** 2 - q2 ** 4 / 12.0))
    slopes.append(math.log(e1 / e2) / math.log(q1 / q2))
print("     余项 log-log 斜率（q = 0.4 至 0.05）：%s（期望 6）" % " ".join("%.4f" % s for s in slopes))
resid = [abs(4.0 * S2([q, 0, 0]) - (q ** 2 - q ** 4 / 12.0)) for q in qs]
print("     余项绝对值：%s（按 q^6 递减，比值应接近 2^6 = 64 或 4^6）"
      % " ".join("%.3e" % v for v in resid))

print()
print("[D2] 方向指纹")
dirs = {'[100]': np.array([1.0, 0, 0]), '[110]': np.array([1, 1, 0]) / math.sqrt(2), '[111]': np.array([1, 1, 1]) / math.sqrt(3)}
vals = {}
for nm, n in dirs.items():
    s4 = float(np.sum(n ** 4))
    vals[nm] = s4
    print("     %s：sum_a n_a^4 = %.10f ｜ 相对破坏系数 = (1/24) x %.6f = %.10f" % (nm, s4, s4, s4 / 24.0))
print("     比值（对 [111] 归一）：%.6f : %.6f : 1（= 6:3:2）"
      % (vals['[100]'] / vals['[111]'], vals['[110]'] / vals['[111]']))
# 球面平均（带 sin(theta) 权重）
th = np.linspace(0, math.pi, 2001)
ph = np.linspace(0, 2 * math.pi, 401)
acc = 0.0
wsum = 0.0
for t in th:
    w = math.sin(t)
    for p in ph:
        n = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])
        acc += w * float(np.sum(n ** 4))
        wsum += w
print("     球面平均 sum_a <n_a^4> = %.10f（期望 3/5 = 0.6）" % (acc / wsum))

print()
print("[D3] 次光速界（无隙模式，布里渊区抽样）")
worst = 0.0
arg = None
for t in np.linspace(0.01, math.pi - 0.01, 16):
    for p in np.linspace(0, 2 * math.pi, 16):
        n = np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])
        for qq in np.linspace(0.02, 3.0, 80):
            qv = n * qq
            r = S1(qv) / (4.0 * S2(qv))
            if r > worst:
                worst, arg = r, (t, p, qq)
print("     抽样最大 |v_g|^2 / c^2 = %.12f（期望 <= 1）；最大点在 q = %.3f" % (worst, arg[2]))
print("     长波极限 q = 1e-6：%.12f（期望 1）" % (S1([1e-6, 0, 0]) / (4.0 * S2([1e-6, 0, 0]))))

print()
print("[D4] 两约定等价（能量用 eV）")
for E_eV, tag in ((1.0, "1 eV"), (1e12, "1 TeV")):
    d1 = (E_eV / EP_t_eV) ** 2 / 24.0
    EP_base_eV = 2 * math.pi * EP_t_eV
    d2 = math.pi ** 2 / 6.0 * (E_eV / EP_base_eV) ** 2
    print("     %s：[100] 约定一 = %.6e ｜ 约定二 = %.6e ｜ 比 %.12f" % (tag, d1, d2, d2 / d1))
print("     E_P^(t) = hbar/t_P = %.6e eV（标准普朗克能量 1.2209e28 eV）" % EP_t_eV)

print()
print("[D5] 登记")
rows = [
    ("其一", "展开", "四阶系数 1/12，余项斜率 6"),
    ("其二", "方向", "sum n^4 = 1、1/2、1/3，比值 6:3:2，球面平均 3/5"),
    ("其三", "界", "|v_g|^2/c^2 <= 1，等号只在长波极限"),
    ("其四", "约定", "两种能标约定给同一个数"),
    ("其五", "落点", "科普第八节补半页 + 本注记"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-6s %s" % (r1, r2, r3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
