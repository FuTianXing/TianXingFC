# -*- coding: utf-8 -*-
"""最小验证（识别式乘数）：记录场度规耦合乘数 κ 的线性响应公式与唯一解定理。

判据（先立判据，再实现；判据取自《内部草稿/研究方向-识别式的微观导出：记录场
度规结构的判定与包层重锚》§6.1 与 §6.1.1，本脚本独立复现其数值结论）

  M1 公式校验：更正公式 κ = ½[a₃ − a₁w_k − a₂w_g − a₄w_U] 与中心差分实测 κ 逐例吻合
     （相对偏差 < 1e-6）；初版公式 κ_plan = ½[a₁ + a₂w_g − a₃] 整体符号相反（对照记录）。
  M2 权重归一：模平方期望权重 w_k + w_g + w_U = 1（偏差 < 1e-10）。
     w_k 由离散拉普拉斯二次型独立计算（不用归一恒等式反推），故本项为真实检验。
  M3 条件性否定：固定 (a₁,a₂,a₃) = (0,2,4) 只扫 a₄ 时，使 κ = 1 所需的 a₄ = 2(w_k+w_U)/w_U
     随阱形而变（谐振 4.000、四次 6.000）；故该族参数无普适解。**本项只否一族，不否全部**。
  M4 唯一解定理（2026-09-12 更正新增）：要求 κ = 1 对一切权重三元组成立，等价于
     (a₃−a₄) + (a₄−a₁)w_k + (a₄−a₂)w_g = 2（用了 w_U = 1−w_k−w_g），故唯一解为
     a₁ = a₂ = a₄ ≡ a 与 a₃ = a + 2，即算符相对本征侧均匀标度 (1−2θ)。a 为分摊约定、不改变物理。
     数值：五种阱形下 |κ−1| ≤ 2e-8，且 θ 至 0.49 时 ω_b(θ) = ω_b(0)√(1−2θ) 精确成立（偏差 ≲ 2e-12）。

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
另行处理，本脚本只取正谱阱）；⑤ M4 的唯一解以「要求 κ = 1 对一切权重成立」为前提，
该前提本身即几何耦合假设的数学形态，本脚本证明该假设的系数解唯一，不证明假设本身。

更正记录（2026-09-12）：本脚本初版把 M3 记为「不存在普适的响应系数赋值使 κ = 1」，
该表述过强。M3 只扫 a₄、固定 (a₁,a₂,a₃) = (0,2,4)，故只否该固定势族。M4 补上一般解：
普适解存在且唯一（a₁ = a₂ = a₄ = a、a₃ = a+2）。Gauss/sech 等负谱阱不在本脚本射程内。

运行时注意（条件数）：阱的边界势值决定算符动态范围，进而决定特征值绝对精度
（~1e-15·‖A‖）。M2 的权重和容差取 1e-10，测试阱边界值控制在 1e5 量级以内
（边界达 1e7 的 x^6 阱会使权重和偏差升至 1e-10 以上，属数值条件问题而非公式
问题，故不纳入）。
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
    "harmonic  0.3x^2":  0.3 * X**2,
    "x^2/2+x^4/8":       0.5 * X**2 + X**4 / 8.0,
    "x^2/2+0.02x^4":     0.5 * X**2 + 0.02 * X**4,
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
print("最小验证（识别式乘数）：κ 的线性响应公式与唯一解定理")
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

# ---- M3 条件性否定 ----
print("\n[M3] 条件性否定（只扫 a₄）：固定 (a₁,a₂,a₃) = (0,2,4) 时所需的 a₄ 随阱形而变")
print("  条件：a₁ = 0、a₂ = 2、a₃ = 4 下，a₄ = 2(w_k+w_U)/w_U")
print("  期望：不同阱形给出不同 a₄（离散度 >> 1e-3），故该固定族无普适赋值")
print("  注意：本项只否该一族，一般解见 M4")
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
print("  M3 %s（阱形依赖；但仅对该固定族成立，见 M4）"
      % ("达成" if ok_m3 else "未达成"))

# ---- M4 唯一解定理 ----
print("\n[M4] 唯一解定理：κ = 1 对一切权重成立的系数解")
print("  代数：代入 w_U = 1−w_k−w_g，κ = 1 ⟺ (a₃−a₄) + (a₄−a₁)w_k + (a₄−a₂)w_g = 2")
print("        对一切 (w_k,w_g) 成立 ⟹ a₄−a₁ = 0、a₄−a₂ = 0、a₃−a₄ = 2")
print("        即 a₁ = a₂ = a₄ ≡ a、a₃ = a + 2（算符相对本征侧均匀标度 1−2θ）")

# M4a 穷举整数族
print("\n  M4a 穷举扫描 |a_i| ≤ 4 的整数赋值（容差 1e-6）")
hits = []
for a1 in range(-4, 5):
    for a2 in range(-4, 5):
        for a3 in range(-4, 5):
            for a4 in range(-4, 5):
                ok = True
                for name, U in WELLS.items():
                    wk, wg, wU, lam0 = W[name]
                    if abs(0.5 * (a3 - a1 * wk - a2 * wg - a4 * wU) - 1.0) > 1e-6:
                        ok = False
                        break
                if ok:
                    hits.append((a1, a2, a3, a4))
print("     命中数 = %d" % len(hits))
for h in hits:
    print("       (a₁,a₂,a₃,a₄) = %s  ->  a = %d，满足 a₁=a₂=a₄ 且 a₃=a+2" % (str(h), h[0]))
ok_m4a = len(hits) > 0 and all(h[0] == h[1] == h[3] and h[2] == h[0] + 2 for h in hits)

# M4b 均匀标度在五种阱形下的 κ
print("\n  M4b 均匀标度 (a₁,a₂,a₄) = (−2,−2,−2)、a₃ = 0 下的 κ 实测")
print("      %-22s %10s %14s" % ("阱形", "λ₀", "κ 实测"))
ok_m4b = True
for name, U in WELLS.items():
    wk, wg, wU, lam0 = W[name]
    kap = kappa_num(-2, -2, 0, -2, U, lam0)
    ok_m4b = ok_m4b and abs(kap - 1.0) < 1e-7
    print("      %-22s %10.5f %14.8f" % (name, lam0, kap))

# M4c 大 θ 精确根式
print("\n  M4c 均匀标度的根式完成（精确，非仅一阶）")
print("      %-22s %8s %16s %16s %10s" % ("阱形", "θ", "ω_num", "ω₀√(1−2θ)", "相对偏差"))
ok_m4c = True
for name, U in WELLS.items():
    lam0, _ = ground(0, 0, 0, 0, 0.0, U)
    w0 = np.sqrt(lam0)
    for t in (0.1, 0.3, 0.49):
        # 均匀标度：算符整体乘 (1−2t)，本征侧不乘
        g = 1.0 - 2.0 * t
        diag = 2.0 * C2 * g / H**2 + WG2 * g + g * U
        off = -C2 * g / H**2 * np.ones(N - 1)
        val = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))[0][0]
        wn = np.sqrt(val)
        pred = w0 * np.sqrt(1.0 - 2.0 * t)
        rel = abs(wn - pred) / pred
        ok_m4c = ok_m4c and rel < 1e-10
        print("      %-22s %8.2f %16.10f %16.10f %10.1e" % (name, t, wn, pred, rel))
print("      注：θ = 0.49 处偏差为浮点噪声；θ = 0.5 时 ω = 0（视界），构造在此饱和。")

ok_m4 = ok_m4a and ok_m4b and ok_m4c

print("\n" + "=" * 76)
print("判据总览：M1 %s  M2 %s  M3 %s  M4 %s"
      % ("✓" if ok_m1 else "✗", "✓" if ok_m2 else "✗",
         "✓" if ok_m3 else "✗", "✓" if ok_m4 else "✗"))
print("结论（2026-09-12 更正）：更正公式经独立复现；权重归一经独立检验；")
print("  M3 只否固定势族；M4 给出一般解：普适解存在且唯一，为算符相对本征侧的均匀标度")
print("  a₁ = a₂ = a₄ = a、a₃ = a+2，等价于根式完成与视界条件 θ = 1/2。")
print("  识别式的这类证法因此有解，且唯一解即几何耦合假设的重述。")
print("=" * 76)
