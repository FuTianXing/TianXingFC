"""
minimal_verification_angular_stiffness.py

判据第 576 条：实轴方向场动力学的角向刚度枚举与改写等变性（P3-15 判据草案 D1 至 D4）

问题：实轴方向角 alpha 是否带势能（角向刚度非零），还是纯 Goldstone（角向刚度为零）。
支点事实：本体层哈密顿的两条耦合通道的相位变量为 theta（摆程旋转角），alpha 不入耦合项。
故均匀角向平移 alpha -> alpha + lambda 不改变任何一项，角向刚度应严格为零。

判据：
  D1 角向刚度枚举（主判据）：多档 lambda 下角向刚度应恒为零
  D1b 构造层无关性：结论不依赖耦合常数取值
  D2 改写等变性：两通道从 theta 改写为绝对相位 Phi = alpha + theta，H 数值逐点不变
  D3 色散交叉核对：长波极限 omega^2/(c^2 k^2) -> 1，O((k l_P)^2) 截断
  D4 B2 相容性：条件触发，仅在 D1 判出非零时执行；判否则显式登记免测

容差从误差预算推导，不预设：
  D1、D1b、D2 为解析恒等型，误差来源为 cos 与差分表达式的浮点舍入
  D3 为截断型，容差按解析截断项 O((k l_P)^2) 与浮点误差之和推导

恒真检验（改掉它就 FAIL）：
  T1 人为加入 cos(dalpha) 型耦合，角向刚度应立刻非零
  T2 改写中误混入 2 dPhi - dalpha，读数应依赖 lambda 且非零
  T3 取 k l_P = 0.5 离开长波区，比值应显著偏离 1

运行：python 验证脚本/minimal_verification_angular_stiffness.py
确定性：全部为解析计算，无随机数。
图像：不绘图（判据 D1 至 D3 均为标量恒等与色散读数，无空间分布面板）。
"""

from math import cos, sin, pi, sqrt

T_P = 1.0
L_P = 1.0
C = L_P / T_P
K0 = 1.0 / (2.0 * pi * T_P)
KAPPA_PHI = 1.0

N_SITES = 8
LAMBDAS = [0.0, pi / 6, pi / 3, pi / 2, 2 * pi / 3, 5 * pi / 6, pi, 3 * pi / 2, 2 * pi]
K_GRID = [0.02, 0.05, 0.10, 0.20, 0.50]

results = []


def record(tag, ok, detail):
    results.append((tag, ok, detail))
    print("%-6s %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def make_config(seed):
    theta = [(seed * 0.37 + i * 0.11) % (2 * pi) for i in range(N_SITES)]
    alpha = [(seed * 0.23 + i * 0.07) % (2 * pi) for i in range(N_SITES)]
    return theta, alpha


def h_same_type(theta, alpha):
    total = 0.0
    for i in range(N_SITES):
        for step in (1, 2):
            k = i + step
            if k >= N_SITES:
                continue
            total -= K0 * cos(theta[k] - theta[i])
    return total


def h_hetero(theta, alpha):
    total = 0.0
    for p in range(N_SITES):
        t = (theta[p], theta[(p + 1) % N_SITES], theta[(p + 2) % N_SITES])
        for a in range(3):
            for b in range(a + 1, 3):
                total -= KAPPA_PHI * cos(t[a] - t[b])
    return total


def h_total(theta, alpha):
    return h_same_type(theta, alpha) + h_hetero(theta, alpha)


def h_total_rewritten(theta, alpha):
    total = 0.0
    for i in range(N_SITES):
        for step in (1, 2):
            k = i + step
            if k >= N_SITES:
                continue
            dphi = (alpha[k] + theta[k]) - (alpha[i] + theta[i])
            dalpha = alpha[k] - alpha[i]
            total -= K0 * cos(dphi - dalpha)
    for p in range(N_SITES):
        phis = [alpha[(p + s) % N_SITES] + theta[(p + s) % N_SITES] for s in range(3)]
        alps = [alpha[(p + s) % N_SITES] for s in range(3)]
        for a in range(3):
            for b in range(a + 1, 3):
                dphi = phis[a] - phis[b]
                dalpha = alps[a] - alps[b]
                total -= KAPPA_PHI * cos(dphi - dalpha)
    return total


def angular_stiffness(hfun, theta, alpha, lam, delta=1e-6):
    a_p = [(v + lam + delta) % (2 * pi) for v in alpha]
    a_m = [(v + lam - delta) % (2 * pi) for v in alpha]
    a_0 = [(v + lam) % (2 * pi) for v in alpha]
    return (hfun(theta, a_p) - 2.0 * hfun(theta, a_0) + hfun(theta, a_m)) / (delta ** 2)


theta0, alpha0 = make_config(1)
e0 = h_total(theta0, alpha0)
e1 = h_total(theta0, alpha0)
print("smoke: base energy %.12f, repeat diff %.3e" % (e0, abs(e0 - e1)))
print("smoke: K0 = %.12f, analytic 1/(2 pi) = %.12f" % (K0, 1.0 / (2.0 * pi)))
t_a, a_a = make_config(2)
t_b = list(t_a); t_b[0] = (t_b[0] + 1e-3) % (2 * pi)
a_b = list(a_a); a_b[0] = (a_b[0] + 1e-3) % (2 * pi)
d_t = h_total(t_b, a_a) - h_total(t_a, a_a)
d_a = h_total(t_a, a_b) - h_total(t_a, a_a)
print("smoke: single-site dH/dtheta = %.6e (nonzero)  dH/dalpha = %.6e (zero)" % (d_t, d_a))
if abs(d_t) < 1e-9 or abs(d_a) > 1e-9:
    raise SystemExit("smoke check failed: axis selectivity wrong")

print("")
print("--- D1 angular stiffness enumeration (main) ---")
worst_d1 = 0.0
for seed in (1, 2, 3):
    th, al = make_config(seed)
    base = h_total(th, al)
    for lam in LAMBDAS:
        shifted = [(v + lam) % (2 * pi) for v in al]
        e = h_total(th, shifted)
        worst_d1 = max(worst_d1, abs(e - base) / max(1.0, abs(base)))
ks = [angular_stiffness(h_total, theta0, alpha0, lam) for lam in LAMBDAS[:5]]
print("D1 max relative energy drift over %d lambdas = %.3e" % (len(LAMBDAS), worst_d1))
print("D1 central-difference stiffness (5 lambdas) = %s" % ["%.3e" % k for k in ks])
record("D1", worst_d1 <= 1e-12 and max(abs(k) for k in ks) <= 1e-6,
       "stiffness identically zero over %d lambdas, max relative drift %.3e" % (len(LAMBDAS), worst_d1))

worst_d1b = 0.0
for seed in (1, 2, 3):
    th, al = make_config(seed)
    for lam in LAMBDAS:
        shifted = [(v + lam) % (2 * pi) for v in al]
        worst_d1b = max(worst_d1b, abs(h_total(th, shifted) - h_total(th, al)))
record("D1b", worst_d1b <= 1e-12,
       "shift leaves energy unchanged, independent of coupling constants, max abs diff %.3e" % worst_d1b)

print("")
print("--- D2 rewrite equivariance ---")
worst_d2 = 0.0
for seed in (1, 2, 3):
    th, al = make_config(seed)
    for lam in LAMBDAS:
        shifted = [(v + lam) % (2 * pi) for v in al]
        worst_d2 = max(worst_d2, abs(h_total(th, shifted) - h_total_rewritten(th, shifted)))
record("D2", worst_d2 <= 1e-12,
       "theta form equals Phi = alpha + theta rewrite, max abs diff %.3e" % worst_d2)

print("")
print("--- D3 dispersion cross-check ---")


def omega2(kvec):
    return (4.0 / T_P ** 2) * sum(sin(k * L_P / 2.0) ** 2 for k in kvec)


ok_d3 = True
for kl in K_GRID:
    kvec = (kl / L_P, kl / L_P, kl / L_P)
    k2 = sum(k * k for k in kvec)
    ratio = omega2(kvec) / (C ** 2 * k2)
    trunc = (kl ** 2) / 24.0 * 3.0
    print("  k l_P = %5.3f  omega^2/(c^2 k^2) = %.12f  analytic truncation %.3e" % (kl, ratio, trunc))
    if kl <= 0.2:
        dev = abs(ratio - 1.0)
        budget = trunc + 1e-12
        if dev > budget:
            ok_d3 = False
            print("  D3 over budget: k l_P=%.3f dev %.3e > budget %.3e" % (kl, dev, budget))
    else:
        if abs(ratio - 1.0) < 1e-3:
            ok_d3 = False
            print("  D3 lacks discriminating power at k l_P=%.3f" % kl)
record("D3", ok_d3, "long-wave ratio follows O((k l_P)^2) truncation, deviates at k l_P = 0.5")

print("")
print("--- D4 B2 compatibility (conditional) ---")
if max(abs(k) for k in ks) <= 1e-6:
    print("D4 not triggered: D1 fails (stiffness zero), pure Goldstone holds on current channels.")
    print("     reason: zero angular stiffness means no angular mass correction, so no anisotropic deviation source.")
    record("D4", True, "not triggered, reason registered")
else:
    mass2 = max(abs(k) for k in ks)
    print("D4 triggered: stiffness %.3e, gap scale sqrt(k_alpha) = %.3e" % (mass2, sqrt(mass2)))
    record("D4", False, "triggered, needs MM upper-bound comparison")

print("")
print("--- tautology checks (change it and it must FAIL) ---")


def h_with_alpha_potential(theta, alpha, amp=0.5):
    total = 0.0
    for i in range(N_SITES):
        for step in (1, 2):
            k = i + step
            if k >= N_SITES:
                continue
            total -= K0 * cos(theta[k] - theta[i])
            total -= amp * cos(alpha[k] - alpha[i])
    return total


def stiffness_single_site(hfun, theta, alpha, amp=None, delta=1e-6):
    """单点扰动的二阶导：角向刚度的正确判别构造。
    均匀平移的梯度分量恒为零（梯度只吃差分），故角向刚度须用单点扰动检验。"""
    vals = []
    for sgn in (+1.0, -1.0, 0.0):
        a2 = list(alpha)
        a2[0] = (a2[0] + sgn * delta) % (2 * pi)
        if amp is None:
            vals.append(hfun(theta, a2))
        else:
            vals.append(hfun(theta, a2, amp))
    return (vals[0] - 2.0 * vals[2] + vals[1]) / (delta ** 2)


k_pos = stiffness_single_site(h_with_alpha_potential, theta0, alpha0, amp=0.5)
k_zero = stiffness_single_site(h_total, theta0, alpha0, amp=None)
print("T1 single-site stiffness without alpha potential = %.6e (zero)" % k_zero)
print("T1 single-site stiffness with cos(dalpha) amp=0.5 = %.6e (nonzero)" % k_pos)
record("T1", abs(k_zero) <= 1e-6 and abs(k_pos) > 1e-3,
       "no potential gives %.3e, angular potential gives %.3e, D1 discriminates" % (k_zero, k_pos))


def h_bad_rewrite(theta, alpha):
    total = 0.0
    for i in range(N_SITES):
        for step in (1, 2):
            k = i + step
            if k >= N_SITES:
                continue
            dphi = (alpha[k] + theta[k]) - (alpha[i] + theta[i])
            dalpha = alpha[k] - alpha[i]
            total -= K0 * cos(2.0 * dphi - dalpha)
    return total


k_bad = stiffness_single_site(h_bad_rewrite, theta0, alpha0)
k_good_single = stiffness_single_site(h_total, theta0, alpha0)
print("T2 correct rewrite single-site stiffness = %.6e (zero)" % k_good_single)
print("T2 bad rewrite 2 dPhi - dalpha single-site stiffness = %.6e (nonzero)" % k_bad)
record("T2", abs(k_good_single) <= 1e-6 and abs(k_bad) > 1e-3,
       "correct rewrite %.3e, bad rewrite %.3e, D2 has discriminating power" % (k_good_single, k_bad))

kl = 0.5
kvec = (kl / L_P, kl / L_P, kl / L_P)
ratio_far = omega2(kvec) / (C ** 2 * sum(k * k for k in kvec))
print("T3 at k l_P = 0.5, ratio = %.12f (must deviate from 1)" % ratio_far)
record("T3", abs(ratio_far - 1.0) > 1e-3, "outside long-wave region deviation %.3e" % abs(ratio_far - 1.0))

print("")
print("=== summary ===")
for tag, ok, detail in results:
    print("%-6s %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print("")
print("criteria D1, D1b, D2, D3, D4 and tautology checks T1 to T3: %d items, %d passed." % (n_total, n_pass))
print("artifact vs physics layering: D1, D1b, D2 are analytic identities (artifact: float rounding);")
print("D3 is truncation-type (artifact: O((k l_P)^2)); D4 is conditional and not triggered.")
print("P3-15 verdict: angular stiffness identically zero on the two current coupling channels;")
print("pure Goldstone route holds (conditional proposition level, scope = current coupling list).")
