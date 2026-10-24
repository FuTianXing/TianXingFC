"""
angular_gradient_potential.py

判据第 577 条：非零角向差的偏转势能与真空稳定性（第 576 条射程收窄后的新增余项）

问题：第 576 条只证「均匀取向族内角向刚度为零」（D1 的均匀平移构造）。
本条测真正的「偏转是否有势能」：相邻格点朝向不一致（Delta alpha != 0）时的能量代价，
以及朝向一致是否为能量最低点（真空稳定性）。

判据：
  D5 非零 Delta alpha 的能量代价：测 H 在非零 Delta alpha 处的读数
  D6 真空稳定性：朝向一致是否为能量最低点（二阶导符号）
  D7 两种嵌入的判别：alpha 是否通过单位圆嵌入改变等效耦合强度
  D8 B2 相容性：条件触发

支点：定理 1 给 Delta alpha = k l_P（波矢 = 实轴方向梯度）。
若 alpha 进入同类耦合，能量应含 cos(Delta theta + Delta alpha) 型项，
等效耦合强度变为 K0/(2 pi)，与库内色散关系 4/t_P^2 sum sin^2(k l_P/2) 冲突。

容差从误差预算推导：
  D5、D6、D7 为解析恒等型，浮点舍入级
  D8 条件触发

恒真检验：
  T1 人为加入 cos(Delta alpha) 型势
  T2 改用 cos(Delta theta + Delta alpha) 嵌入

运行：python 验证脚本/angular_gradient_potential.py
确定性：解析计算，无随机数。
图像：不绘图（标量能量读数，无空间分布面板）。
"""

from math import cos, sin, pi, sqrt

T_P = 1.0
L_P = 1.0
C = L_P / T_P
K0 = 1.0 / (2.0 * pi * T_P)
KAPPA_PHI = 1.0

N_SITES = 8
DELTA_LIST = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
results = []


def record(tag, ok, detail):
    results.append((tag, ok, detail))
    print("%-6s %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def make_config(seed):
    theta = [(seed * 0.37 + i * 0.11) % (2 * pi) for i in range(N_SITES)]
    alpha = [(seed * 0.23 + i * 0.07) % (2 * pi) for i in range(N_SITES)]
    return theta, alpha


def make_gradient_config(delta):
    """朝向不一致构型：alpha 沿 x 方向线性递增，相邻差恒为 delta。"""
    theta = [(0.37 + i * 0.11) % (2 * pi) for i in range(N_SITES)]
    alpha = [(delta * i) % (2 * pi) for i in range(N_SITES)]
    return theta, alpha


def h_total(theta, alpha, embed=False, amp=0.0):
    """本体哈密顿。
    embed=False：同类耦合作用于 Delta theta（库内写法，alpha 不入）
    embed=True ：同类耦合作用于 Delta Phi = Delta theta + Delta alpha（单位圆嵌入写法）
    amp > 0   ：额外加入 cos(Delta alpha) 型角向势（供恒真检验）
    """
    total = 0.0
    for i in range(N_SITES):
        for step in (1, 2):
            k = i + step
            if k >= N_SITES:
                continue
            dt = theta[k] - theta[i]
            da = alpha[k] - alpha[i]
            arg = dt + da if embed else dt
            total -= K0 * cos(arg)
            if amp != 0.0:
                total -= amp * cos(da)
    for p in range(N_SITES):
        t = (theta[p], theta[(p + 1) % N_SITES], theta[(p + 2) % N_SITES])
        for a in range(3):
            for b in range(a + 1, 3):
                total -= KAPPA_PHI * cos(t[a] - t[b])
    return total


def second_derivative(f, x, delta=1e-5):
    return (f(x + delta) - 2.0 * f(x) + f(x - delta)) / (delta ** 2)


print("smoke: K0 = %.12f, analytic 1/(2 pi) = %.12f" % (K0, 1.0 / (2.0 * pi)))
th0, al0 = make_config(1)
base = h_total(th0, al0)
print("smoke: base energy %.12f" % base)

# ============ D5 非零 Delta alpha 的能量代价 ============
print("")
print("--- D5 energy cost at nonzero Delta alpha ---")
costs = []
for d in DELTA_LIST:
    th, al = make_gradient_config(d)
    e = h_total(th, al)
    costs.append((d, e))
    print("  Delta alpha = %4.2f  H = %+.12f  dH = %+.3e" % (d, e, e - base))

worst_d5 = max(abs(e - base) for _, e in costs)
record("D5", worst_d5 <= 1e-12,
       "nonzero Delta alpha leaves H unchanged, max |dH| = %.3e" % worst_d5)

# ============ D6 真空稳定性：朝向一致是否为能量最低 ============
print("")
print("--- D6 vacuum stability (is uniform orientation a minimum?) ---")
d2_zero = second_derivative(lambda d: h_total(*make_gradient_config(d)), 0.0)
print("D6 d2H/d(delta)^2 at delta=0 = %+.6e" % d2_zero)
record("D6", abs(d2_zero) <= 1e-6,
       "curvature at uniform orientation = %.3e (flat, neither min nor max)" % d2_zero)

# ============ D7 两种嵌入的判别 ============
print("")
print("--- D7 two embeddings (does alpha change effective coupling?) ---")
th, al = make_gradient_config(0.7)
e_plain = h_total(th, al, embed=False)
e_embed = h_total(th, al, embed=True)
print("D7 plain  (Delta theta)          H = %+.12f" % e_plain)
print("D7 embedded (Delta theta + Delta alpha) H = %+.12f" % e_embed)
diff = abs(e_plain - e_embed)
print("D7 |plain - embedded| = %.6e" % diff)

# 若库内色散关系成立，则库内写法为 plain；嵌入写法将改变等效耦合 K0 -> K0/(2 pi)
keff_plain = K0
keff_embed = K0 / (2.0 * pi)
print("D7 effective coupling: plain %.6f vs embedded %.6f, ratio %.6f" % (keff_plain, keff_embed, keff_plain / keff_embed))

# 判别：两写法在非零 delta 下给出不同读数，故判据有区分力
record("D7", diff > 1e-6,
       "two embeddings differ by %.3e, D7 discriminates embedding choice" % diff)

# ============ D8 B2 相容性（条件触发） ============
print("")
print("--- D8 B2 compatibility (conditional) ---")
if worst_d5 <= 1e-12 and abs(d2_zero) <= 1e-6:
    print("D8 not triggered: no angular gradient cost, no vacuum instability, B2 has no deviation source.")
    record("D8", True, "not triggered, reason registered")
else:
    print("D8 triggered: nonzero cost or curvature at uniform orientation.")
    record("D8", False, "triggered, needs MM comparison")

# ============ 恒真检验 ============
print("")
print("--- tautology checks ---")


def h_with_alpha_potential(theta, alpha, amp):
    return h_total(theta, alpha, embed=False, amp=amp)


d = 1e-5
vals = []
for sgn in (+1.0, -1.0, 0.0):
    dd = 0.0 + sgn * d
    th, al = make_gradient_config(dd)
    vals.append(h_with_alpha_potential(th, al, 0.5))
k_amp = (vals[0] - 2.0 * vals[2] + vals[1]) / (d ** 2)
print("T1 with cos(Delta alpha) amp=0.5: d2H/d(delta)^2 = %+.6e (nonzero)" % k_amp)
record("T1", abs(k_amp) > 1e-3, "angular potential gives nonzero curvature %.3e, D6 discriminates" % k_amp)

k_embed_curv = second_derivative(lambda x: h_total(*make_gradient_config(x), embed=True), 0.0)
print("T2 embedded writing: d2H/d(delta)^2 at 0 = %+.6e" % k_embed_curv)
record("T2", abs(k_embed_curv) > 1e-3,
       "embedded writing gives curvature %.3e, differs from plain %.3e" % (k_embed_curv, d2_zero))

print("")
print("=== summary ===")
for tag, ok, detail in results:
    print("%-6s %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print("")
print("criteria D5, D6, D7, D8 and tautology checks T1, T2: %d items, %d passed." % (n_total, n_pass))
print("verdict: at nonzero Delta alpha the energy is flat (no angular gradient cost);")
print("uniform orientation is flat, not a strict minimum. Scope: current coupling list only.")
print("P3-15 remains open on: whether uniform orientation is a true vacuum (dynamical stability),")
print("which requires a dynamics for alpha beyond the static Hamiltonian.")
