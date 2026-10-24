# -*- coding: utf-8 -*-
r"""
write_rate_residence_channel.py

写入率接驻留判据的通道合成轮：受激率能否作为驻留判据的独立通道。

来源：作者 2026-09-21 指令（同意执行）。输入：第 292、295、304 条的受激率形式 dot N = A v g_激，
g_激 = min(1, w/(v Dt))；C1 线的驻留读数（C1-4 的驻留阈值 K >= 0.24 至 0.25、C1-3 的小 K 塌缩）。

判据（先行）：
  R1 两段形式的恒等：分界点 v_min = w/Dt（扫描 v 验证 min(1, w/(vDt)) 的切换位置，偏差 0）
  R2 灵敏度表：五档分箱宽下的 v_min/c，给出「线性档覆盖的速度区间」
  R3 驻留判据的合成：两通道（能量型锁定读数 × 写入率读数）的四格判定表
  R4 与 C1-4 窗口的对照：锁定窗内受激项应归零、塌缩档应非零；指出需要补的新读数
"""
import numpy as np

W_LATT = 15.8
C_UNIT = 1.0
rows = []

# R1 分界点恒等
Dt = 10.0
v_min = W_LATT / Dt
vs = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 1.58, 1.6, 2.0, 5.0, 20.0])
g = np.minimum(1.0, W_LATT / (vs * Dt))
v_lo = float(np.max(vs[g >= 1.0]))
v_hi = float(np.min(vs[g < 1.0]))
bracketed = (v_lo <= v_min + 1e-12) and (v_hi > v_min)
rows.append(("R1", "切换点 v = w/Dt = %.3f 落在扫描区间的夹逼内（%.2f, %.2f）" % (v_min, v_lo, v_hi), "夹逼成立" if bracketed else "夹逼失败", "v_lo <= v_min < v_hi", bracketed))

# R2 灵敏度表
DTS = [1.0, 10.0, 1e3, 1e6, 1e12]
tab = [(Dt_, W_LATT / Dt_) for Dt_ in DTS]
rows.append(("R2", "五档分箱的 v_min/c（w = %.1f l_P）" % W_LATT, " / ".join("%.3g" % x[1] for x in tab), "登记", True))
cover = [x for x in tab if x[1] >= C_UNIT]
rows.append(("R2b", "线性档覆盖全部物理速度的分箱上限（v_min >= c）", "Dt <= %.1f t_P" % (W_LATT / C_UNIT), "登记", True))

# R3 四格判定表
grid = [
 ("能量通道判锁定（K >= 0.24）", "受激项 = 0（v = 0，缺陷驻留）", "驻留"),
 ("能量通道判锁定，写入率显著高于本底", "矛盾格：需查本底标定或登记口径", "待查"),
 ("能量通道判未锁定（K <= 0.20）", "受激项 > 0（缺陷滑移或塌缩，v != 0）", "非驻留"),
 ("能量通道判未锁定，写入率等于本底", "矛盾格：缺陷被别处钉住（边界或缺陷相互作用）", "待查"),
]
rows.append(("R3", "四格判定表", "见表", "登记", True))

# R4 与 C1-4 对照的数值形式
K_lock = 0.24
K_small = 0.005
rows.append(("R4", "锁定窗与塌缩档的受激率形式", "锁定窗（K >= %.2f）：v = 0 给受激项 0；塌缩档（K = %.3f）：v > 0 待测" % (K_lock, K_small), "登记（需补塌缩档速度读数）", True))

print("=== 写入率接驻留判据通道合成判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-48s 实测 %-50s 期望 %-24s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("R1 细节：w = %.1f l_P，Dt = %.1f，v_min = %.3f（单位 c）；g_激 在 v <= v_min 时为 1（线性档），v > v_min 时为 w/(vDt)（饱和档）" % (W_LATT, Dt, v_min))
print("R2 灵敏度（分箱宽 -> v_min）：")
for Dt_, vm in tab:
    print("   Dt = %.3g t_P -> v_min = %.3g c ；该档线性可测的速度上界" % (Dt_, vm))
print("R3 四格判定表：")
for a, b, c in grid:
    print("   %-34s | %-40s -> %s" % (a, b, c))
print("R4：C1-4 的驻留阈值 K >= %.2f 属能量型判据（格点内锁定）；写入率通道测的是速度，二者独立。" % K_lock)
print("    在锁定窗内 v = 0，受激项严格为零，写入率退到本底；在 K <= 0.20 的塌缩档，缺陷滑移或塌缩，v != 0，受激项非零。")
print("    要把它做成对照读数，缺的是塌缩档的速度读数（C1-3 只给 E(R) 标度与塌缩命运，未给速度）。")
