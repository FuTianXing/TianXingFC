# -*- coding: utf-8 -*-
"""最小验证（识别式乘数）：记录场度规耦合乘数 κ 的线性响应公式与否定性结果。

判据（先立判据，再实现；判据取自《内部草稿/研究方向-识别式的微观导出：记录场
度规结构的判定与包层重锚》§6.1 与 §6.1.1，本脚本独立复现其数值结论）

  M1 公式校验：更正公式 κ = ½[a₃ − a₁w_k − a₂w_g − a₄w_U] 与中心差分实测 κ 逐例吻合
     （相对偏差 < 1e-6）；初版公式 κ_plan = ½[a₁ + a₂w_g − a₃] 整体符号相反（对照记录）。
  M2 权重归一：模平方期望权重 w_k + w_g + w_U = 1（偏差 < 1e-10）。
     w_k 由离散拉普拉斯二次型独立计算（不用归一恒等式反推），故本项为真实检验。
  M3 否定性结果：使 κ = 1 所需的 a₄ = 2(w_k+w_U)/w_U 随阱形而变（谐振 4.000、四次 6.000），
     离散度 >> 容差；故不存在普适的响应系数赋值使 κ = 1。

模型设定：一维束缚态有效算符（路径乙，度规型）
  [−c²(1+a₁θ)∂ₓ² + ω_g²(1+a₂θ) + (1+a₄θ)U(x)] ψ = (1+a₃θ) ω_b² ψ
  固有时比率 ω_b(θ)/ω_b(0) = 1 − κθ + O(θ²)，κ = −ω_b'(0)/ω_b(0)。
  数值：N = 4000 均匀格点、h = 2L/N（L = 30）、三对角精确对角化（确定性，无随机迭代）。

实测 vs 期望：逐例打印 κ_num / κ_formula / κ_plan（初版）与 a₄ 需求表。
运行：在工作区根目录执行 `python 验证脚本/minimal_verification_identification_multiplier.py`。
本脚本为判据核验，不产图。

不验证：① 记录场与束缚态算符的耦合是否独立于背景格距（4.4 剩余义务）；
② T0 度规结构判定（由 S6 完成值裁决，本脚本不裁决）；③ 三维（本装置为一维，
参照复合位错纲领 C1 的三维位错形态）；④ 负谱阱（λ₀ < 0 时 ω₀ 虚，线性化失效，
另行处理，本脚本只取正谱阱）。
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

C2 = 1.0
WG2 = 1.0
L, N = 30.0, 4000
H = 2.0 * L / N
X = np.linspace(-L + H, L - H, N)

WELLS = {
    "harmonic  U=x^2/2": 0.5 * X**2,
    "quartic   U=x^4/4": X**4 / 4.0,
}

CASES = [(0, 2, 4, 0), (0, 2, 4, 2), (1, 0, -2, 0),
         (0, 0, -2, 0), (1, 2, 4, 3), (-1, 0.5, 0.25, 0)]


def lap(psi):
    """离散拉普拉斯 D2 psi（三对角，端点单侧）。"""
    t = np.empty_like(psi)
    t[1:-1] = (psi[:-2] - 2.0 * psi[1:-1] + psi[2:]) / H**2
    t[0] = (-2.0 * psi[0] + psi[1]) / H**2
    t[-1] = (psi[-2] - 2.0 * psi[-1]) / H**2
    return t


def ground(a1, a2, a3, a4, t, U):
    """最低本征值 λ = ω_b² 与归一化本征向量。"""
    g = 1.0 + a1 * t
    diag = 2.0 * C2 * g / H**2 + WG2 * (1.0 + a2 * t) + (1.0 + a4 * t) * U
    off = -C2 * g / H**2 * np.ones(N - 1)
    val, vec = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
    psi = vec[:, 0]
    lam = val[0] / (1.0 + a3 * t)
    return lam, psi


def weights(U):
    """模平方期望权重：(w_k, w_g, w_U, λ₀)。w_k 由拉普拉斯二次型独立计算。"""
    lam0, psi = ground(0, 0, 0, 0, 0.0, U)
    n2 = psi @ psi
    wk = -C2 * (psi @ lap(psi)) / n2 / lam0
    wg = WG2 / lam0
    wU = (psi @ (U * psi)) / n2 / lam0
    return wk, wg, wU, lam0


def kappa_num(a1, a2, a3, a4, U, lam0):
    """中心差分实测 κ = −ω_b'(0)/ω_b(0)。"""
    dt = 1e-5
    lp, _ = ground(a1, a2, a3, a4, dt, U)
    lm, _ = ground(a1, a2, a3, a4, -dt, U)
    om0 = np.sqrt(lam0)
    return -((np.sqrt(lp) - np.sqrt(lm)) / (2.0 * dt)) / om0


print("=" * 76)
print("最小验证（识别式乘数）：κ 的线性响应公式与否定性结果")
print("=" * 76)

# ---- M2 权重归一 ----
print("\n[M2] 权重归一（w_k 由离散拉普拉斯二次型独立计算）")
print("  期望：w_k + w_g + w_U = 1，偏差 < 1e-10")
ok_m2 = True
W = {}
print("  %-24s %10s %10s %10s %9s %12s" % ("阱形", "w_k", "w_g", "w_U", "求和", "λ₀"))
for name, U in WELLS.items():
    wk, wg, wU, lam0 = weights(U)
    W[name] = (wk, wg, wU, lam0)
    s = wk + wg + wU
    ok_m2 = ok_m2 and abs(s - 1.0) < 1e-10
    print("  %-24s %10.5f %10.5f %10.5f %9.1e %12.6f" % (name, wk, wg, wU, s - 1.0, lam0))
print("  M2 %s（实测 vs 期望：求和偏差见上列最后第二栏）"
      % ("达成" if ok_m2 else "未达成"))

# ---- M1 公式校验 ----
print("\n[M1] 公式校验（谐振阱，中心差分 dt = 1e-5）")
print("  更正式：κ = ½[a₃ − a₁w_k − a₂w_g − a₄w_U]（期望：与实测吻合，相对偏差 < 1e-6）")
print("  初版式：κ_plan = ½[a₁ + a₂w_g − a₃]（对照：符号相反、权重错配）")
wk, wg, wU, lam0 = W["harmonic  U=x^2/2"]
U = WELLS["harmonic  U=x^2/2"]
print("  %-22s %13s %13s %13s %10s" % ("(a1,a2,a3,a4)", "κ_num", "κ_formula", "κ_plan", "相对偏差"))
ok_m1 = True
for (a1, a2, a3, a4) in CASES:
    kn = kappa_num(a1, a2, a3, a4, U, lam0)
    kf = 0.5 * (a3 - a1 * wk - a2 * wg - a4 * wU)
    kp = 0.5 * (a1 + a2 * wg - a3)
    rel = abs(kn - kf) / max(abs(kf), 1e-30)
    ok_m1 = ok_m1 and rel < 1e-6
    print("  %-22s %13.8f %13.8f %13.8f %10.1e"
          % (str((a1, a2, a3, a4)), kn, kf, kp, rel))
print("  M1 %s" % ("达成" if ok_m1 else "未达成"))

# ---- M3 否定性结果 ----
print("\n[M3] 否定性结果：使 κ = 1 所需的 a₄ 随阱形而变")
print("  条件：a₁ = 0、a₂ = 2、a₃ = 4 下，a₄ = 2(w_k+w_U)/w_U")
print("  期望：不同阱形给出不同 a₄（离散度 >> 1e-3），故不存在普适赋值")
print("  %-24s %10s %12s %14s" % ("阱形", "w_k/w_U", "a₄ 需求", "该 a₄ 下 κ 实测"))
a4s = []
for name, U in WELLS.items():
    wk, wg, wU, lam0 = W[name]
    a4 = 2.0 * (wk + wU) / wU
    kap = kappa_num(0, 2, 4, a4, U, lam0)
    a4s.append(a4)
    print("  %-24s %10.5f %12.5f %14.6f" % (name, wk / wU, a4, kap))
spread = max(a4s) / min(a4s) - 1.0
ok_m3 = spread > 1e-3
print("  a₄ 需求离散度 = %.5f（%.4f 对 %.4f）" % (spread, a4s[0], a4s[1]))
print("  M3 %s（阱形依赖 = κ 依赖时钟构造 = K4 失败签名）"
      % ("达成" if ok_m3 else "未达成"))

print("\n" + "=" * 76)
print("判据总览：M1 %s  M2 %s  M3 %s"
      % ("✓" if ok_m1 else "✗", "✓" if ok_m2 else "✗", "✓" if ok_m3 else "✗"))
print("结论：更正公式经独立复现；权重归一经独立检验；不存在普适 (a₁,a₂,a₃,a₄) 使 κ = 1。")
print("=" * 76)
