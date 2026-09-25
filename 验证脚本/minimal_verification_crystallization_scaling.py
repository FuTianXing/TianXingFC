# -*- coding: utf-8 -*-
"""结晶扩张相变的标度清点：标准临界标度全判负与 3/5 的机制域（第 415 条，P0-5 候选机制三轮）。

判据：
  W1 标准临界动力学类的 (nu, z) 对照表（对照允许带）
  W2 进带所需的 (nu, z) 域（z 扫描与 nu 窗口）
  W3 体系候选指数（1/3、1/2、1）的判定与交点复算
  W4 等效输运律：带内 alpha 对应 D_eff 的增长指数
  W5 清点表汇总

运行：python 验证脚本/minimal_verification_crystallization_scaling.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "crystallization_scaling_audit.png")

F_OBS = 2.12e-105
T_EQ = 2.745e55
Z_EQ = 3400.0
W_DEF = 3.80e21
T_UNIV = 4.35e17 / 5.391247e-44

C_CURVE = (Z_EQ ** -3.0 / F_OBS) ** (1.0 / 3.0)
D_REQ = C_CURVE ** 2 / T_EQ
A_MAX = math.log(W_DEF) / math.log(T_EQ * (W_DEF / C_CURVE) ** 2)
A_MIN = 0.5 + math.log(C_CURVE / math.sqrt(T_EQ)) / math.log(T_UNIV)
LO, HI = A_MIN, A_MAX

def t_of_alpha(alpha):
    if abs(alpha - 0.5) < 1e-12:
        return None
    return (C_CURVE / math.sqrt(T_EQ)) ** (1.0 / (alpha - 0.5))

def nu_of_alpha(alpha, z):
    # alpha = nu/(1+nu z)  ->  nu = alpha/(1-alpha z)
    den = 1.0 - alpha * z
    if den <= 0.0:
        return None
    return alpha / den

print("== 框架（与第 413/414 条同式复算） ==")
print("  C = %.4e，D_req = %.4e 格^2/拍" % (C_CURVE, D_REQ))
print("  允许带：[%.4f, %.4f]" % (LO, HI))
print()

print("== W1 标准临界动力学类对照 ==")
classes = [
    ("3D Ising (model A)", 0.6300, 2.024),
    ("3D XY (model A)", 0.6717, 2.024),
    ("3D Heisenberg (model A)", 0.7112, 2.024),
    ("平均场 (model A)", 0.5, 2.0),
    ("平均场守恒 (model B)", 0.5, 4.0),
]
for name, nu, z in classes:
    alpha = nu / (1.0 + nu * z)
    ok = LO <= alpha <= HI
    print("  %s：nu = %.4f，z = %.3f，alpha = %.4f，带内 = %s" % (name, nu, z, alpha, ok))
print("  KT 类（nu 趋于无穷，z = 2）：alpha = 1/z = %.4f，带内 = %s" % (0.5, False))
print("  通则：z >= 2 时 alpha <= 1/2 < %.4f（nu 趋于无穷取上确界 1/z），扩散型动力学全体判负" % LO)
print()

print("== W2 进带所需的 (nu, z) 域 ==")
print("  条件：alpha = nu/(1+nu z) 落在 [%.4f, %.4f]" % (LO, HI))
for z in (0.5, 1.0, 1.5, 5.0/3.0, 1.786, 2.0):
    a_hi = 1.0 / z
    nu_lo = nu_of_alpha(LO, z)
    nu_hi = nu_of_alpha(HI, z)
    if a_hi < LO:
        print("  z = %.3f：alpha 上确界 1/z = %.4f < %.4f，不可达" % (z, a_hi, LO))
    elif nu_lo is None or nu_hi is None or (nu_hi is not None and nu_hi <= nu_lo):
        print("  z = %.3f：alpha 上确界 1/z = %.4f；带内仅下端可达，nu >= %.3f（上端不可达）" % (z, a_hi, nu_lo if nu_lo else float('nan')))
    else:
        print("  z = %.3f：带内可达，nu 窗口 = [%.3f, %.3f]（上确界 1/z = %.4f）" % (z, nu_lo, nu_hi, a_hi))
print("  边界：z = %.4f 时上确界恰为带上端 %.4f；z = %.4f 时恰为带下端 %.4f" % (1.0/HI, HI, 1.0/LO, LO))
print("  汇总：带内可达要求 z < %.4f；z = 1 需 nu 在 [%.3f, %.3f]，z = 1.5 需 nu 在 [%.3f, %.3f]" % (
    1.0/LO, nu_of_alpha(LO, 1.0), nu_of_alpha(HI, 1.0), nu_of_alpha(LO, 1.5), nu_of_alpha(HI, 1.5)))
print()

print("== W3 体系候选指数判定 ==")
t13 = t_of_alpha(1.0/3.0)
print("  alpha = 1/3（KZ 幂律）：t = %.3e 拍（< 1 拍，非物理），判负" % t13)
print("  alpha = 1/2（追赶变体）：与链条曲线重合（无定点；等价于 D = D_req = %.4e），退化为单参数曲线" % D_REQ)
t1 = t_of_alpha(1.0)
print("  alpha = 1（视界线性）：t = %.3e 拍，xi/w_def = %.3e，判负（第 413 条 X2 复算）" % (t1, t1/W_DEF))
print()

print("== W4 等效输运律（D_eff = xi^2/t = t^{2 alpha - 1}） ==")
for alpha, tag in ((0.5, "扩散（曲线）"), (3.0/5.0, "3/5 候选"), (LO, "带下端"), (HI, "带上端")):
    print("  alpha = %.4f（%s）：D_eff 增长指数 = %.4f" % (alpha, tag, 2.0*alpha - 1.0))
print("  表述：带内 alpha 对应输运系数的缓慢幂律增长，指数区间 [%.4f, %.4f]" % (2*LO-1.0, 2*HI-1.0))
print()

print("== W5 清点表 ==")
rows = [
    ("临界点结构", "零动力不动点（原点全静止）", "连续相变临界点不成立（第 381 条 V1）"),
    ("临界减速", "无（校准谱 kappa k^2 无临界结构）", "Zurek 标度律前提不成立"),
    ("吸引性", "交错基态无吸引性", "无序到有序的弛豫图像不成立"),
    ("体系机制", "种子注入 + 多核成核 + 波前扩张 + 接缝", "畴尺度与过渡宽度分离（第 381 条 V2）"),
    ("畴尺度候选", "KZ 幂律 19.5 / 追赶 60.8 / 视界 7.4e3 步", "均在带外判负或退化（本轮 W3）"),
    ("3D 标准类", "alpha <= 0.29", "全体判负（本轮 W1）"),
    ("KT 类", "前提缺失 + z = 2 给 0.5", "判负（本轮 W1/W2）"),
    ("剩余机制域", "z < %.4f 的动力学 或 成核-接缝几何生长" % (1.0/LO), "两条待建模（移交）"),
]
for a, b, c in rows:
    print("  %s | %s | %s" % (a, b, c))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
zs = [0.3 + i*0.02 for i in range(0, 96)]
a_hi = [1.0/z for z in zs]
ax.plot(zs, a_hi, color="#c0392b", lw=2.0, label="alpha_max = 1/z (nu -> inf)")
ax.axhspan(LO, HI, color="#ffe9b0", alpha=0.85, label="allowed band [%.4f, %.4f]" % (LO, HI))
ax.axvline(1.0/HI, color="#888888", ls=":", lw=1.2)
ax.axvline(1.0/LO, color="#888888", ls=":", lw=1.2)
ax.annotate("z = %.4f" % (1.0/HI), (1.0/HI, 0.72), fontsize=9, ha="right")
ax.annotate("z = %.4f" % (1.0/LO), (1.0/LO, 0.72), fontsize=9, ha="left")
ax.plot([2.024], [0.5], "o", color="#14508c", markersize=7)
ax.annotate("3D classes (z=2.024): alpha <= 0.29", (2.024, 0.5), textcoords="offset points", xytext=(-80, 14), fontsize=9, color="#14508c")
ax.set_xlim(0.3, 2.2)
ax.set_ylim(0.2, 1.05)
ax.set_xlabel("dynamic exponent z", fontsize=10.5)
ax.set_ylabel("alpha ceiling", fontsize=10.5)
ax.set_title("W2 band reachability: z < %.3f required" % (1.0/LO), fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper right")

ax = axes[1]
z_fill = [0.3 + i*0.005 for i in range(0, int((1.0/LO - 0.3)/0.005))]
nu_lo_f = [nu_of_alpha(LO, z) for z in z_fill]
nu_hi_f = [nu_of_alpha(HI, z) if nu_of_alpha(HI, z) is not None else float("nan") for z in z_fill]
ax.fill_between(z_fill, nu_lo_f, [min(v, 12.0) for v in nu_hi_f], color="#b8e0b8", alpha=0.9, label="in-band (nu, z)")
ax.plot([1.0], [1.2717], "o", color="#2f7a2f", markersize=7)
ax.annotate("z = 1: nu in [%.2f, %.2f]" % (nu_of_alpha(LO,1.0), nu_of_alpha(HI,1.0)), (1.0, 1.39), textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.plot([1.5], [3.492], "o", color="#2f7a2f", markersize=7)
ax.annotate("z = 1.5: nu in [%.2f, %.2f]" % (nu_of_alpha(LO,1.5), nu_of_alpha(HI,1.5)), (1.5, 3.49), textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xlim(0.3, 1.85)
ax.set_ylim(0, 12)
ax.set_xlabel("dynamic exponent z", fontsize=10.5)
ax.set_ylabel("critical exponent nu", fontsize=10.5)
ax.set_title("W2 in-band region in (nu, z)", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper right")

fig.savefig(IMG)
print("  已保存：", IMG)
