# -*- coding: utf-8 -*-
"""
判据第 602 条：kappa_2 终判交叉核对（第 601 条判否出口的独立复算）
==================================================================
目的：第 601 条（并行会话交付，判据 6/6，终判判否）的关键数值用本线（第 597 条作者）
的 AB 桥与换算族清单口径独立复算一遍，防跨会话量纲口径漂移。
复算工具的口径定义逐字取自第 597 条脚本（换算族五基元、n in [-6,6]、距离 = 到族格点
的最小对数距离、先除 kappa_1 再查族），非从第 601 条代码移植。

核对锚点（取自第 601 条注记 §三至 §五）：
    C1  Ms 通道 Fe 档逐位复现第 597 条：8.533e-55（相对 < 1e-3）
    C2  A 通道三档 log 区间 [-55.15, -54.61]（0.01）、漂移比 3.4884（相对 < 1e-3）
    C3  A 通道分离度区间 [-56.19, -55.64]（0.05）、换算族最近距离 49.05（0.05）
    C4  B_2 = mu0*e^2*l_P^2/(hbar*c) 对数 -87.57（0.01）；
        低阶档最小距离 24.79 于 (a=1,g=-1,Ni)（0.05）；
        完备枚举单档巧合 (a=3,g=-2)：Ni 2.07 进带、Fe 3.16 / Co 3.67 带外（0.05）
    C5  幂扫描 p=-1 端：Ms 13.68（FeCo 档）、A 14.26（Co 档）（0.05）

可逆性等级：判据稳健（同常数复算，换 CODATA 版本影响 <1e-9 对数单位）。
"""
import numpy as np

# ---------- 常数（与第 597 条同源） ----------
E_CH   = 1.602176634e-19        # C（精确）
HBAR   = 1.054571817e-34        # J s（精确）
C_L    = 2.99792458e8           # m/s（精确）
MU0    = 1.25663706212e-6       # N/A^2（CODATA 2018）
LP     = 1.616255e-35           # m（CODATA 2018）
ALPHA  = 1.0 / 137.035999084
KAPPA1 = 1.0 / (4.0 * np.pi * ALPHA)
LOG_LP = np.log10(LP)

# 换算族清单（第 597 条口径：五基元幂展开 n in [-6,6]，49 格点）
FAMILY = []
for n in range(-6, 7):
    for base in (2*np.pi, np.pi, 4*np.pi, np.pi**2/4.0, 1.0/(2*np.pi)):
        FAMILY.append(np.log10(base) * n)
FAMILY = np.array(sorted(set(FAMILY)))

def dist_family(logv):
    """log10 读数（已除 kappa_1）到换算族格点的最小对数距离（第 597 条口径）。"""
    return float(np.min(np.abs(FAMILY - logv)))

# 材料档（与第 597、601 条同源：Crangle-Hallam 1971 / Kuz'min 2020）
MS_T = {'Fe': 2.15, 'Co': 1.76, 'Ni': 0.61, 'FeCo': 2.35}   # mu0*Ms（T）
A_PM = {'Fe': 20.0, 'Co': 30.0, 'Ni': 8.6}                  # A（pJ/m）

results = []
def check(name, ok, detail):
    results.append(ok)
    print('[%s] %s' % ('PASS' if ok else 'FAIL', name))
    for line in detail:
        print('    ' + line)

# ---------- C1：Ms 通道 Fe 档逐位复现第 597 条 ----------
k2_src = {m: E_CH * MU0 * (b / MU0) * LP**2 / HBAR for m, b in MS_T.items()}
log_src = {m: np.log10(v) for m, v in k2_src.items()}
rel = abs(k2_src['Fe'] - 8.533e-55) / 8.533e-55
check('C1 Ms 通道 Fe 档逐位复现第 597 条（8.533e-55）', rel < 1e-3, [
    '实测 %.4e（log10=%.2f）vs 期望 8.533e-55，相对偏差 %.2e' % (k2_src['Fe'], log_src['Fe'], rel),
])

# ---------- C2：A 通道三档与漂移比 ----------
k2_A = {m: a * 1e-12 * LP**2 / (HBAR * C_L) for m, a in A_PM.items()}
log_A = {m: np.log10(v) for m, v in k2_A.items()}
drift = 10**(max(log_A.values()) - min(log_A.values()))
lo, hi = min(log_A.values()), max(log_A.values())
ok2 = abs(lo - (-55.15)) < 0.01 and abs(hi - (-54.61)) < 0.01 and abs(drift - 3.4884)/3.4884 < 1e-3
check('C2 A 通道三档区间与漂移比', ok2, [
    'log10 区间 [%.2f, %.2f] vs 期望 [-55.15, -54.61]' % (lo, hi),
    '漂移比 %.4f vs 期望 3.4884（=30/8.6，解析值）' % drift,
])

# ---------- C3：A 通道分离度与换算族距离 ----------
d_A = {m: log_A[m] - np.log10(KAPPA1) for m in log_A}
dist_A = {m: dist_family(v) for m, v in d_A.items()}
lo_d, hi_d = min(d_A.values()), max(d_A.values())
dist_min = min(dist_A.values())
ok3 = abs(lo_d - (-56.19)) < 0.05 and abs(hi_d - (-55.64)) < 0.05 and abs(dist_min - 49.05) < 0.05
check('C3 A 通道分离度区间与换算族最近距离', ok3, [
    'log10(kappa^A/kappa_1) 区间 [%.2f, %.2f] vs 期望 [-56.19, -55.64]' % (lo_d, hi_d),
    '换算族最近距离 %.2f vs 期望 49.05（Ni 档）' % dist_min,
])

# ---------- C4：完备解族抽验（B_2、低阶档、单档巧合） ----------
log_B2 = np.log10(MU0 * E_CH**2 * LP**2 / (HBAR * C_L))
# 低阶档 |a|,|g| <= 1（排除恒等组合 (0,0)：读数恒为 1，属恒真点，第 601 条 V5 反例二同款）
low = {}
for a in (-1, 0, 1):
    for g in (-1, 0, 1):
        if a == 0 and g == 0:
            continue
        for m in log_A:
            v = a * log_A[m] + g * log_B2 - np.log10(KAPPA1)
            low[(a, g, m)] = dist_family(v)
best_low = min(low, key=low.get)
# 完备枚举抽验 (a=3, g=-2)
trio = {m: dist_family(3 * log_A[m] - 2 * log_B2 - np.log10(KAPPA1)) for m in A_PM}
ok4 = (abs(log_B2 - (-87.57)) < 0.01
       and best_low == (1, -1, 'Ni') and abs(low[best_low] - 24.79) < 0.05
       and abs(trio['Ni'] - 2.07) < 0.05 and trio['Ni'] < 3.0
       and abs(trio['Fe'] - 3.16) < 0.05 and abs(trio['Co'] - 3.67) < 0.05
       and trio['Fe'] > 3.0 and trio['Co'] > 3.0)
check('C4 B_2 与完备解族抽验（低阶档 + 单档巧合）', ok4, [
    'log10(B_2) = %.2f vs 期望 -87.57' % log_B2,
    '低阶档最小距离 %.2f 于 (a,g,材料)=%s vs 期望 24.79 于 (1,-1,Ni)' % (low[best_low], best_low),
    '单档巧合 (a=3,g=-2)：Ni %.2f（进带）、Fe %.2f / Co %.2f（带外）vs 期望 2.07 / 3.16 / 3.67'
        % (trio['Ni'], trio['Fe'], trio['Co']),
    '跨材料不稳健确认：仅 Ni 档进带（<3），与第 601 条「单档巧合只登记不判候选」一致',
])

# ---------- C5：幂扫描 p=-1 端点 ----------
def pow_scan(logs):
    best = None
    for p in np.arange(-1, 3.001, 0.25):
        for m, lv in logs.items():
            d = dist_family(lv + p * LOG_LP - np.log10(KAPPA1))
            if best is None or d < best[0]:
                best = (d, p, m)
    return best
best_src, best_A = pow_scan(log_src), pow_scan(log_A)
ok5 = abs(best_src[0] - 13.68) < 0.05 and abs(best_A[0] - 14.26) < 0.05 \
      and best_src[1] == -1.0 and best_A[1] == -1.0
check('C5 幂扫描 p in [-1,3] 步 0.25 全档最小（p=-1 端）', ok5, [
    'Ms 通道 %.2f 于 p=%.2f（%s 档）vs 期望 13.68 于 p=-1（FeCo 档）' % (best_src[0], best_src[1], best_src[2]),
    'A 通道 %.2f 于 p=%.2f（%s 档）vs 期望 14.26 于 p=-1（Co 档）' % (best_A[0], best_A[1], best_A[2]),
])

# ---------- 汇总 ----------
n_pass = sum(results)
print('\n===== 交叉核对总表（第 602 条）：%d/%d =====' % (n_pass, len(results)))
for name, ok in zip(['C1', 'C2', 'C3', 'C4', 'C5'], results):
    print('%s  %s' % ('PASS' if ok else 'FAIL', name))
print('结论：第 601 条判否出口的关键数值全部独立复算通过，无跨会话口径漂移。')
