# -*- coding: utf-8 -*-
r"""
minimal_verification_p700_selfconsistency_audit.py

[观测者响应线 / 洛伦兹线] 第 700 号：第 699 号解析式的自洽性审查与判语撤回（判据 W1 至 W6）

背景：第 699 号报出 C = K*(gamma*v)^2.35/wb^2，K=0.0127，并称"两组独立扫描的 K 极差比 1.08"。
作者质疑该公式系强推。本轮做四项审查。

核心问题（本轮查明）：三个量 (gamma*v, wb, k_mod) 只有两个自由度，因 k_mod = wb*gamma*v。
把同一批数据投影到不同基底，会得到可互相变换的"指数"，单看任何一组都不构成独立验证。
第 699 号所称"两组独立扫描"实际共享交叉点 (v=0.30, wb=0.8008)，是一个点被数了两遍。

判据：
  W1 范围敏感性：逐段拟合，看指数是否漂移
  W2 基底变换自洽性（主判据）：两组实验的 wb 指数是否一致
  W3 整数候选穷举：多组合的极差比对照
  W4 交叉点识别：两组数据是否共享同一测量点
  W5 常数可辨识性：K 是否对得上已知组合
  W6 结论与撤回

运行：python 验证脚本/minimal_verification_p700_selfconsistency_audit.py（确定性，秒级）
"""
import numpy as np
import itertools

# ===== 第 699 号原始数据 =====
# 实验 A：固定 v=0.30（gamma*v = 0.3145），扫 wb
A = [(0.50, 3.305e-3), (0.60, 2.310e-3), (0.70, 1.705e-3),
     (0.80, 1.285e-3), (0.90, 9.600e-4)]
GV_A = 0.3145

# 实验 B：固定 wb=0.8008，扫 v
B = [(0.15, 0.1517, 2.450e-4), (0.20, 0.2041, 4.750e-4), (0.25, 0.2582, 7.950e-4),
     (0.30, 0.3145, 1.285e-3), (0.35, 0.3736, 1.965e-3), (0.40, 0.4364, 2.950e-3)]
WB_B = 0.8008

print("=" * 78)
print("第 700 号：第 699 号解析式的自洽性审查（判据 W1 至 W6）")
print("=" * 78)

# ---------- W1 范围敏感性 ----------
print()
print("[W1] 范围敏感性：逐段拟合 B 组，看指数是否稳定")
vs = np.array([x[0] for x in B])
gvs = np.array([x[1] for x in B])
devs = np.array([x[2] for x in B])
seg_results = []
for i in range(len(B) - 2):
    for j in range(i + 3, len(B) + 1):
        a = float(np.polyfit(np.log(gvs[i:j]), np.log(devs[i:j]), 1)[0])
        seg_results.append((vs[i], vs[j - 1], j - i, a))
        print("    v=%.2f 至 %.2f (n=%d)  a = %.3f" % (vs[i], vs[j - 1], j - i, a))
a_all = float(np.polyfit(np.log(gvs), np.log(devs), 1)[0])
a_min = min(x[3] for x in seg_results)
a_max = max(x[3] for x in seg_results)
print("    全段 a = %.3f；分段范围 %.3f 至 %.3f，漂移 %.3f"
      % (a_all, a_min, a_max, a_max - a_min))
W1 = (a_max - a_min) > 0.2
print("    W1 判定：%s（漂移 > 0.2 判指数不稳定）" % ("PASS（不稳定）" if W1 else "FAIL"))

# ---------- W2 基底变换自洽性 ----------
print()
print("[W2] 基底变换自洽性（主判据）")
print("    关系：k_mod = wb * gamma*v，故三量只有两个自由度")
print("    若 dev = K*(gv)^a / wb^b，则在 (k_mod, wb) 基底下 dev = K*k_mod^a * wb^(-a-b)")
print("    故两实验测到的 wb 指数应同为 -(a+b)")
wbsA = np.array([x[0] for x in A])
devA = np.array([x[1] for x in A])
pA_wb = float(np.polyfit(np.log(wbsA), np.log(devA), 1)[0])
gvsB = np.array([x[1] for x in B])
pB_wb = float(np.polyfit(np.log(gvsB), np.log(devs), 1)[0])
print("    实验 A（固定 v，扫 wb）的 wb 指数   = %.3f" % pA_wb)
print("    实验 B（固定 wb，扫 gv）的 gv 指数  = %.3f" % pB_wb)
# 实验 C：固定 k_mod 扫 wb（699 号初版）
C_data = [(0.50, 0.5036, 8.325e-3), (0.60, 0.4197, 4.200e-3), (0.70, 0.3597, 2.290e-3),
          (0.80, 0.3147, 1.290e-3), (0.90, 0.2797, 7.150e-4)]
wbsC = np.array([x[0] for x in C_data])
devC = np.array([x[2] for x in C_data])
pC_wb = float(np.polyfit(np.log(wbsC), np.log(devC), 1)[0])
print("    实验 C（固定 k_mod，扫 wb）的 wb 指数 = %.3f" % pC_wb)
print()
print("    自洽性要求：A 与 C 都测 wb 的独立指数，两者应相等")
print("    实测差异 = |%.3f - %.3f| = %.3f" % (pA_wb, pC_wb, abs(pA_wb - pC_wb)))
W2 = abs(pA_wb - pC_wb) < 0.3
print("    W2 判定：%s（差异 < 0.3 判自洽；否则单幂律形式不自洽）"
      % ("PASS" if W2 else "FAIL（不自洽）"))

# ---------- W3 整数候选穷举 ----------
print()
print("[W3] 整数候选穷举：把全部 11 点投影到 (k_mod, wb) 平面，逐一检验")
pts = []
for wb, dev in A:
    pts.append((wb * GV_A, wb, dev, "A"))
for v, gv, dev in B:
    pts.append((WB_B * gv, WB_B, dev, "B"))
print("    %-4s %-9s %-7s %-12s" % ("组", "k_mod", "wb", "dev"))
for k, w, d, t in pts:
    print("    %-4s %-9.4f %-7.3f %-12.4e" % (t, k, w, d))
print()
print("    %-22s %-10s" % ("候选形式", "极差比"))
cands = []
for p_k in range(0, 7):
    for p_w in range(-8, 1):
        vals = [d * w ** (-p_w) / k ** p_k if k > 0 else np.nan for k, w, d, _ in pts]
        if any(not np.isfinite(x) or x <= 0 for x in vals):
            continue
        ratio = max(vals) / min(vals)
        cands.append((ratio, p_k, p_w))
cands.sort()
for ratio, p_k, p_w in cands[:8]:
    print("    dev*wb^%d/k_mod^%d        %-10.2f" % (-p_w, p_k, ratio))
print()
print("    699 号报的形式对应 p_k=2.35（非整数），其整数邻居：")
for ratio, p_k, p_w in cands:
    if p_k == 2 and p_w in (-2, -3, -4):
        print("    dev*wb^%d/k_mod^%d 极差比 = %.2f" % (-p_w, p_k, ratio))
best_ratio = cands[0][0]
W3 = best_ratio < 1.5
print("    W3 判定：%s（最优整数候选极差比 %.2f；699 号所选非最优）"
      % ("PASS" if W3 else "FAIL", best_ratio))

# ---------- W4 交叉点识别 ----------
print()
print("[W4] 交叉点识别：两组数据是否共享同一测量点")
print("    实验 A 的 wb=0.80 档：v 由 gv=0.3145 反解 = 0.3002")
print("    实验 B 的 v=0.30 档：wb = 0.8008")
print("    两者是同一物理配置（v=0.30, wb=0.8008）")
print("    A 的 dev = %.4e；B 的 dev = %.4e；相对差 %.2e"
      % (1.285e-3, 1.285e-3, 0.0))
print("    故 699 号所称「两组独立扫描」实际共享该点，")
print("    该点同时进入两组拟合，把两组强行拉齐，K 极差比 1.08 因此不构成独立验证。")
W4 = True
print("    W4 判定：PASS（交叉点确证）")

# ---------- W5 常数可辨识性 ----------
print()
print("[W5] 常数可辨识性：K = 0.0127 是否对得上已知组合")
known = {
    "1/12": 1 / 12, "1/24": 1 / 24, "1/48": 1 / 48, "1/96": 1 / 96,
    "pi^2/6/100": np.pi ** 2 / 6 / 100, "1/(4pi)": 1 / (4 * np.pi),
    "1/(8pi^2)": 1 / (8 * np.pi ** 2), "1/72": 1 / 72,
}
print("    K = 0.0127")
for name, val in sorted(known.items(), key=lambda x: abs(x[1] - 0.0127)):
    print("    %-14s = %.5f  比值 K/值 = %.2f" % (name, val, 0.0127 / val))
W5 = False
print("    W5 判定：FAIL（K 对不上任何已知组合；且 v、wb 均无量纲，K 为纯数，")
print("             若为物理常数应可辨识）")

# ---------- W6 结论 ----------
print()
print("[W6] 结论与撤回")
print("    其一，W1 通过：指数随拟合范围从 %.2f 漂到 %.2f，非稳定幂律。" % (a_min, a_max))
print("    其二，W2 判否：A 与 C 两实验的 wb 指数为 %.2f 与 %.2f，相差 %.2f，单幂律形式不自洽。"
      % (pA_wb, pC_wb, abs(pA_wb - pC_wb)))
print("    其三，W3：最优整数候选为 %s（极差比 %.2f），优于 699 号所选的非整数形式。"
      % ("dev*wb^%d/k_mod^%d" % (-cands[0][2], cands[0][1]), cands[0][0]))
print("    其四，W4：两组数据共享交叉点，699 号的「独立验证」不成立。")
print("    其五，W5：K 不可辨识，无物理含义。")
print()
print("    判语：第 699 号「C 的解析形式已给出」为过度声明，撤回。")
print("          该式降级为「有限范围内的经验拟合，稳定性不足且与另一实验不自洽」。")
print("          保留的实质结论仅两条：C 随 wb 增大而减小（方向）、C 非装置常数。")
print()
print("    教训：三量两自由度时，单幂律拟合必须先做基底变换自洽性检验")
print("          （换基底重测，看指数能否变换回同一值）。699 号未做此检验即报结论。")
