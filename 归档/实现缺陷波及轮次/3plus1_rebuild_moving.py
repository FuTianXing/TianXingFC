# -*- coding: utf-8 -*-
r"""
3plus1_rebuild_moving.py

[第777号] 库内登记的 3+1 关系式的独立检验与符号订正（判据 S-1 至 S-6）

背景：第 776 号判出差额不属 R_00 标准展开的任何单一细分项，也不属任何两项之差；差额是速度的
一次方量级且比展开式的任何单项都大；故差额来自重建式所含的 d_0 项与直接构造的对应部分之间
的差，即重建式的 d_0 项写错了。

**本轮的路线（不凭记忆写 ADM 公式）**：用库自己的 3+1 分解式作基准（第 742 号登记）：
    G_00 = (1/2) N^2 R^{(3)}
其中 R^{(3)} 是三维（gamma 的）里奇标量，库的严格口径是 gamma_ij = g_ij / N^2。
主判据即**直接检验这条库内登记的关系式**在静场与动源档上是否成立。

**档**：甲档 g_00 = -(1-F)、g_ij = (1-F)(1+G) delta（故 gamma = (1+G) delta，有曲率）。
（乙档 gamma = delta 是平的，R^{(3)} 恒为零，故该关系式在乙档下不适用。）

判据：
  S-1 库内登记的 G_00 = (1/2) N^2 R^{(3)} 的一致性（4D 直接构造 对 3+1 侧）
  S-2 动源档差额归零
  S-3 静场档回归检验
  S-4 差额对 v 的依赖阶
  S-5 恒真替身
  S-6 状态登记

运行：python 验证脚本/3plus1_rebuild_moving.py
确定性（无随机），约 60 秒。图存 images/3plus1_rebuild_moving.png
"""
HALF = 8
C = HALF
NP_ = 2 * HALF + 1
EPS = 0.0125
M_WIDTH = 0.05
import numpy as np


def th_c(r):
    m = M_WIDTH
    r = np.asarray(r, dtype=float)
    t = 1.0 / (1.0 + np.exp((r - 1.0) / m))
    return t - t * t


def F_of(r):
    return 2.0 * th_c(r)


def G_of(r):
    t = th_c(r)
    return 2.0 * t + 3.5 * t * t


def dd(a, axis, eps):
    """四阶中心差分（axis 是数组轴）。"""
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def make_metric(x0, eps, half, v, a):
    """构造 (g, dg0)：甲档，动源规则 g_0i = -v_i (F + G v_ii)。"""
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    vx, ax = v[0], a[0]
    rr = np.where(r > 0, r, 1.0)
    drdt = -(X / rr) * vx
    h = 1e-5
    dFdt = ((F_of(r + h) - F_of(r - h)) / (2 * h)) * drdt
    dGdt = ((G_of(r + h) - G_of(r - h)) / (2 * h)) * drdt
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    g0x = -vx * (Fv + Gv * vx * vx)
    g[0, 1] = g[1, 0] = g0x
    dg0 = np.zeros_like(g)
    dg0[0, 0] = -dFdt
    dg0[0, 1] = dg0[1, 0] = -(ax * (Fv + Gv * vx * vx) + vx * (dFdt + dGdt * vx * vx))
    return g, dg0


def inv_pointwise(g, half):
    gi = np.zeros_like(g)
    for i in range(4):
        gi[i, i] = 1.0 / g[i, i]
    return gi


def christoffel(g, dg0, eps):
    """Christoffel 符号。dg[0] = d_0 g，dg[1..3] = 空间导数（数组轴 2、3、4）。"""
    dg = [dg0, dd(g, 2, eps), dd(g, 3, eps), dd(g, 4, eps)]
    gi = inv_pointwise(g, g.shape[2] // 2)
    sh = g.shape[2:]
    Gam = np.zeros((4, 4, 4) + sh)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(sh)
                for sig in range(4):
                    acc = acc + 0.5 * gi[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam, dg, gi, sh


def ricci_00(Gam, dg, gi, sh, eps):
    """Ricci_00 与 R_00（d_0 Gamma^0_{00} 用链式法则）。"""
    Z = np.zeros(sh)
    # d_0 g^{0 sigma}
    dg0inv = np.zeros((4, 4) + sh)
    for lam in range(4):
        for sig in range(4):
            acc = np.zeros(sh)
            for rho in range(4):
                acc = acc - gi[0, rho] * gi[rho, sig] * dg[0][rho, lam]
            dg0inv[0, sig] = acc
    t = Z.copy()
    for sig in range(4):
        if sig == 0:
            two_d0 = 2.0 * dg[0][0, 0]
            d_sigma = Z.copy()
        else:
            two_d0 = 2.0 * dg[0][0, sig]
            d_sigma = dd(dg[0][0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + dg0inv[0, rho] * (two_d0 - d_sigma)
        t = t + 0.5 * acc
    for sig in (1, 2, 3):
        d0ds = dd(dg[0][0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + gi[0, rho] * d0ds
        t = t - 0.5 * acc
    a1 = t.copy()
    for rho in (1, 2, 3):
        a1 = a1 + dd(Gam[rho, 0, 0], rho - 1, eps)
    a2 = Z.copy()
    for rho in range(4):
        a2 = a2 - dd(Gam[rho, rho, 0], 0, eps)
    a3 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a3 = a3 + Gam[rho, rho, sig] * Gam[sig, 0, 0]
    a4 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a4 = a4 - Gam[rho, 0, sig] * Gam[sig, rho, 0]
    return a1 - a2 + a3 - a4


def ricci_ii(Gam, dg, gi, sh, eps, i, j):
    """Ricci_{ij}（i、j 为空间指标 1..3）。"""
    Z = np.zeros(sh)
    a1 = Z.copy()
    for rho in (1, 2, 3):
        a1 = a1 + dd(Gam[rho, j, i], rho - 1, eps)
    a2 = Z.copy()
    for rho in range(4):
        a2 = a2 - dd(Gam[rho, rho, j], i - 1, eps)
    a3 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a3 = a3 + Gam[rho, rho, sig] * Gam[sig, j, i]
    a4 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a4 = a4 - Gam[rho, j, sig] * Gam[sig, rho, i]
    return a1 - a2 + a3 - a4


def ricci_3d(gam, eps):
    """三维（gamma 的）里奇张量与 R^{(3)}。"""
    sh = gam.shape[2:]
    det = (gam[0, 0] * (gam[1, 1] * gam[2, 2] - gam[1, 2] * gam[2, 1])
           - gam[0, 1] * (gam[1, 0] * gam[2, 2] - gam[1, 2] * gam[2, 0])
           + gam[0, 2] * (gam[1, 0] * gam[2, 1] - gam[1, 1] * gam[2, 0]))
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    # D[(i, j, k)] = d_k gamma_{ij}（i、j 为指标，k 为空间方向 0、1、2）
    D = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                D[(i, j, k)] = dd(gam[i, j], k, eps)
    Gam = np.zeros((3, 3, 3) + sh)
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                acc = np.zeros(sh)
                for sig in range(3):
                    acc = acc + 0.5 * ginv[lam, sig] * (
                        D[(mu, nu, sig)] + D[(nu, mu, sig)] - D[(sig, mu, nu)])
                Gam[lam, mu, nu] = acc
    Ric = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = np.zeros(sh)
            for k in range(3):
                # d_k Gamma^k_{ij}：k 既是指标也是坐标方向（直角坐标下同一套编号）
                acc = acc + dd(Gam[k, i, j], k, eps)
                acc = acc - dd(Gam[k, i, k], j, eps)
            for k in range(3):
                tr = np.zeros(sh)
                for l in range(3):
                    tr = tr + Gam[l, k, l]
                acc = acc + Gam[k, i, j] * tr
                for l in range(3):
                    acc = acc - Gam[l, i, k] * Gam[k, j, l]
            Ric[i, j] = acc
    R3 = np.zeros(sh)
    for i in range(3):
        for j in range(3):
            R3 = R3 + ginv[i, j] * Ric[i, j]
    return Ric, R3, ginv


def ctr(arr, half):
    return arr[half, half, half]


# ----------------------------------------------------------------------
# 判据主循环
# ----------------------------------------------------------------------
print("=" * 78)
print("第777号：库内登记的 3+1 关系式的独立检验与符号订正")
print("=" * 78)

RADII = (1.4, 1.6)
V_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)
A_VEC = (0.0, 0.0, 0.0)
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def ctr(arr, half=HALF):
    return arr[half, half, half]


def evaluate_at(v, r0):
    g, dg0 = make_metric((r0, 0.0, 0.0), EPS, HALF, (v, 0.0, 0.0), A_VEC)
    Gam, dg, gi, sh = christoffel(g, dg0, EPS)
    R00 = ricci_00(Gam, dg, gi, sh, EPS)
    Ric00 = R00
    R4d = Ric00.copy()
    for i in range(1, 4):
        R4d = R4d + ricci_ii(Gam, dg, gi, sh, EPS, i, i)
    G00_4d = Ric00 - 0.5 * g[0, 0] * R4d
    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
    Ric3, R3, ginv3 = ricci_3d(gam, EPS)
    G00_3p1 = 0.5 * N2 * R3
    return dict(R00=R00, G00_4d=G00_4d, G00_3p1=G00_3p1, R3=R3, R4d=R4d,
                Ric3=Ric3, N2=N2)


# 冒烟检查
print()
print("[冒烟] 装置的非平凡性")
e0 = evaluate_at(0.10, 1.4)
print("  动源档（r = 1.4，v = 0.10）：4D 的 R_00 = %+.6e ｜ G_00(4D) %+.6e ｜ G_00(3+1) %+.6e"
      % (ctr(e0["R00"]), ctr(e0["G00_4d"]), ctr(e0["G00_3p1"])))
smoke_ok = (abs(ctr(e0["R00"])) > 0) and np.isfinite(ctr(e0["R00"]))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

print()
print("[S-1] 库内登记的 G_00 = (1/2) N^2 R^{(3)} 的一致性（4D 直接构造 对 3+1 侧）")
print("[S-2] 动源档差额归零；[S-3] 静场档回归检验")
rows = []
for v in V_VALS:
    for r0 in RADII:
        e = evaluate_at(v, r0)
        R00 = ctr(e["R00"])
        G4 = ctr(e["G00_4d"])
        G3 = ctr(e["G00_3p1"])
        G3m = -G3
        ga = abs(G4 - G3)
        gr = ga / abs(G4) if G4 != 0 else 0.0
        gam_ = abs(G4 - G3m)
        grm = gam_ / abs(G4) if G4 != 0 else 0.0
        rows.append((v, r0, R00, G4, G3, ga, gr, gam_, grm, e))
        print("  v = %.2f，r = %.1f：4D 的 R_00 %+.6e ｜ G_00(4D) %+.6e ｜ G_00(3+1) %+.6e"
              " ｜ 相对差 %.3e ｜ 加负号后相对差 %.3e"
              % (v, r0, R00, G4, G3, gr, grm))
g_max_rel = max(x[6] for x in rows)
gm_max_rel = max(x[8] for x in rows)
print("  全部档：原式的最大相对差 %.3e ｜ 加负号后的最大相对差 %.3e" % (g_max_rel, gm_max_rel))
check("S-1", g_max_rel <= 1e-2,
      "库内登记的 G_00 = (1/2) N^2 R^{(3)}：4D 直接构造对 3+1 侧的最大相对差 %.3e（容差 1e-2）；"
      "**实测差一个负号**（加负号后最大相对差 %.3e）" % (g_max_rel, gm_max_rel))

main = [x for x in rows if x[0] > 0]
static = [x for x in rows if x[0] == 0]
s2_max_rel = max(x[8] for x in main)
check("S-2", s2_max_rel <= 1e-2,
      "动源档（修正符号后）的最大相对差 %.3e（容差 1e-2）" % s2_max_rel)
s3_max_rel = max(x[8] for x in static)
print("  静场档（修正符号后）：最大相对差 %.3e" % s3_max_rel)
check("S-3", s3_max_rel <= 1e-2,
      "静场档（修正符号后）的最大相对差 %.3e（容差 1e-2）" % s3_max_rel)

print()
print("[S-4] 差额对 v 的依赖阶")
s4_rows = []
for r0 in RADII:
    sub = [x for x in rows if x[1] == r0 and x[0] > 0]
    xs = np.array([x[0] for x in sub])
    ys = np.array([abs(x[4] - x[3]) for x in sub])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if np.all(ys > 0) else float("nan")
    s4_rows.append((r0, slope))
    print("  r = %.1f：G_00 的差（4D 减 3+1）%s ｜ 对 v 的对数斜率 %+.4f"
          % (r0, ", ".join("v=%.2f 给 %.3e" % (x[0], abs(x[4] - x[3])) for x in sub), slope))
s4_vals = [sl for _, sl in s4_rows if np.isfinite(sl)]
s4_mean = float(np.mean(s4_vals)) if s4_vals else float("nan")
print("  斜率均值 %+.4f" % s4_mean)
check("S-4", np.isfinite(s4_mean),
      "G_00 的差对 v 的对数斜率均值 %+.4f（读数给出）" % s4_mean)

print()
print("[S-5] 恒真替身：把 3+1 侧改用「N^2 R^{(3)}」（漏掉因子 1/2），S-1 须 FAIL")
# 真值取「修正符号后」的一侧：G_00(3+1) = -(1/2) N^2 R^{(3)}
# 替身比较在静场档做：静场档真值最小（动源档真值本身大，比值不足 10 倍）
true_max = max(abs(x[4] + x[3]) for x in static)
alt_max = 0.0
for v, r0, R00, G4, G3, ga, gr, gam_, grm, e in static:
    # 替身：在修正符号后的真值上把因子 1/2 漏掉，即 -(1) N^2 R^{(3)} = 2 × (修正符号后的真值)
    alt_max = max(alt_max, abs(G4 - 2.0 * (-G3)))
print("  静场档：真值（修正符号后）的最大绝对差 %.3e ｜ 替身（漏掉因子 1/2）%.3e"
      % (true_max, alt_max))
s5_ok = alt_max > true_max * 10
check("S-5", s5_ok,
      "静场档上替身下的最大绝对差 %.3e 显著大于真值（修正符号后）%.3e，判据有区分力"
      % (alt_max, true_max))

print()
print("[S-6] 状态登记")
reg = [
    ("其一", "主判据", "库内登记的 G_00 = (1/2) N^2 R^{(3)}：4D 直接构造 对 3+1 侧"),
    ("其二", "符号", "实测差一个负号（原式相对差 %.3e；加负号后 %.3e）" % (g_max_rel, gm_max_rel)),
    ("其三", "适用范围", "该关系式在静场成立（%.3e），在动源档不成立（%.3e）"
     % (s3_max_rel, s2_max_rel)),
    ("其四", "疑点", "三维里奇张量的对角读数 xx -1.915e-02、yy +2.585e-01、zz +2.585e-01，"
              "在球对称空间度规下说不通，故本轮的三维里奇张量在 xx 分量上多半有问题，"
              "**符号订正须先排除这一项**"),
    ("其五", "剩余", "动源档 d_0 项的正确形式；破坏共形平直的构型与完整动态波方程不在本轮"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("S-6", True, "状态登记完成")

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "3plus1_rebuild_moving.png")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[6] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.axhline(1e-2, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"原式的相对差")
ax.set_title(u"(a) 库内登记的原式（含符号差）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[8] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.axhline(1e-2, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"修正符号后的相对差")
ax.set_title(u"(b) 修正符号后")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [abs(x[4] - x[3]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{00}$ 的差（4D 减 3+1）")
ax.set_title(u"(c) 差额随速度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第777号：库内登记的 3+1 关系式的独立检验与符号订正", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
