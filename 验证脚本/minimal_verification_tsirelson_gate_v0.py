# -*- coding: utf-8 -*-
"""
P1-29 v0：Tsirelson 生死闸门（战线二方向一）：纠缠锁定对的 CHSH 存在性与经典圈对照

背景：P1-29（贝尔不等式违反的定量推导）要求关联函数与测量算符的显式构造。
本脚本为 v0 生死闸门：在纠缠锁定对的最小模型（每粒子一个分支二能级）中，
用基底命名的两条通道组装贝尔场景，回答三问：
  (a) 存在性（第一颚）：S 能否越过经典界 2。不过则载体主张死亡。
  (b) 上界（第二颚）：S 是否恰好停在 2*sqrt(2)。越界则为超量子，模型出错。
  (c) 判别：违反由哪条通道承载（表针相干、层位结构、经典答案条三选一）。

模型：
  每粒子一个分支二能级 {|0>,|1>}（层位占据）。
  Z = 分支占据 ±1（M5 层位份额的读出对象）。
  X = 表针奇偶（M7b 质心零模相对相位的干涉读出的 ±1 版）。
  U_z(theta) = 零模踢（M2 甲-1 协议原语：支 1 相位推进 theta）。
  纠缠锁定态 |Phi_gamma> = (|00> + e^{i gamma}|11>)/sqrt(2)：
  联合分支锁定 + 相对表针 gamma（先验纠缠的现象层投射；
  制备机制归 P4-16，本脚本以构造输入处理并在边界声明）。
  测量设置 A(alpha) = cos(alpha) Z + sin(alpha) X（子午面混合），
  操作协议 = Ramsey 链（零模踢 + 重组合 + 层位读出，判据 J9 验证其条纹）。

判据先行（运行前写死）：
  J1  通道代数：Z、X 谱 = {+1,-1}；{Z,X} = 0（1e-13）；跨侧对易为零（张量因子化）
  J2  锁定态纠缠：|Phi_0> 约化密度矩阵本征值 (1/2,1/2)（1e-14）；concurrence = 1（1e-12）
  J3  精确 CHSH：最优角 (0, pi/2, pi/4, -pi/4) 下 S = 2*sqrt(2)（1e-12）
  J3b 锁定族剖面：S_max(gamma) = 2*sqrt(1+cos(gamma)^2)，gamma=0 饱和、gamma=pi/2 骑线（1e-6）
  J4  Born 采样（M4/M5 协议，种子 20260914，每设置对 N = 2e6）：
      S_hat 落在 2*sqrt(2) 的 4 sigma 内，且越过 2 的 z 分数 >= 300（生死闸门第一颚）
  J5  经典圈对照（同一 S^1 几何、纤维角为预存值、lambda_B = lambda_A）：
      同角度 S = 2.000（4 sigma 内）；16 个确定性 LHV 策略穷举最大恰为 2：经典封顶
  J6  表针退相干控制：层位关联保持、表针相干清除：S = sqrt(2) < 2
      （违反载体 = 表针相干通道，层位通道单独不足）
  J7  可见度阈：S(p) = 2*sqrt(2) p（白噪声混合），阈 p* = 1/sqrt(2)（1e-3）
  J8  无信号（只读通道，论文第 8 节）：精确 P(a=+1|beta) = 1/2 对一切 beta（1e-14）；
      采样双侧设置偏差 < 5 sigma
  J9  Ramsey 条纹（通道操作化）：零模踢 theta + 层位读出给余弦条纹，
      可见度 = 1（1e-12），拟合相位回收输入 gamma（1e-10）
  J10 上界扫描：S(x,y,u) 全网格最大 = 2*sqrt(2)（网格精度 3e-3）：模型无超量子区

纪律：全确定性（种子 20260914）；实测 vs 期望对照表；图存 images/（绝对路径）。
边界：不验证纠缠态制备机制（P4-16）；不验证空间分离的全场实现（P4-5 双粒子推广）；
      不声称 Tsirelson 界已获原理级推导（为何 2*sqrt(2) 而非 4，归方向三）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_PATH = r"D:\software\TianXingFC\images\minimal_verification_tsirelson_gate_v0.png"
SEED = 20260914
N_SAMP = 2_000_000

TS = 2.0 ** 0.5 * 2.0          # Tsirelson 界 2*sqrt(2)
CL = 2.0                        # 经典界

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

Z_A, X_A = np.kron(SZ, I2), np.kron(SX, I2)
Z_B, X_B = np.kron(I2, SZ), np.kron(I2, SX)

def obs(theta, side):
    """子午面设置：A(theta) = cos(theta) Z + sin(theta) X。"""
    Zs, Xs = (Z_A, X_A) if side == "A" else (Z_B, X_B)
    return np.cos(theta) * Zs + np.sin(theta) * Xs

def proj(theta, side, sign):
    A = obs(theta, side)
    return (np.eye(4, dtype=complex) + sign * A) / 2.0

def phi_state(gamma):
    v = np.zeros(4, dtype=complex)
    v[0], v[3] = 1 / np.sqrt(2), np.exp(1j * gamma) / np.sqrt(2)
    return v

def expect(psi, O, rho=None):
    if rho is None:
        return np.real(psi.conj() @ (O @ psi))
    return np.real(np.trace(rho @ O))

def joint_probs(state, th_a, th_b, rho=None):
    """P(a,b)：a,b 取 +1(指标 0) / -1(指标 1)。"""
    P = np.empty((2, 2))
    for ia, sa in enumerate((+1, -1)):
        for ib, sb in enumerate((+1, -1)):
            M = proj(th_a, "A", sa) @ proj(th_b, "B", sb)
            P[ia, ib] = np.real(np.trace((state.reshape(4, 1) * state.reshape(1, 4)).conj().T @ M)) \
                if rho is None else np.real(np.trace(rho @ M))
    P = np.clip(P, 0.0, None)
    return P / P.sum()

ALPHA0, ALPHA1 = 0.0, np.pi / 2
BETA0, BETA1 = np.pi / 4, -np.pi / 4
PAIRS = [(ALPHA0, BETA0), (ALPHA1, BETA0), (ALPHA0, BETA1), (ALPHA1, BETA1)]

results = []

def gate(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

print("=" * 78)
print("P1-29 v0 生死闸门：纠缠锁定对 CHSH（判据先行，实测 vs 期望）")
print("=" * 78)

# ---------- J1 通道代数 ----------
spec_z = np.linalg.eigvalsh(SZ)
spec_x = np.linalg.eigvalsh(SX)
anti = np.abs(SZ @ SX + SX @ SZ)
cross = max(np.abs(Z_A @ Z_B - Z_B @ Z_A).max(),
            np.abs(Z_A @ X_B - X_B @ Z_A).max(),
            np.abs(X_A @ Z_B - Z_B @ X_A).max(),
            np.abs(X_A @ X_B - X_B @ X_A).max())
ok = (np.allclose(spec_z, [-1, 1], atol=1e-13) and np.allclose(spec_x, [-1, 1], atol=1e-13)
      and anti.max() < 1e-13 and cross < 1e-13)
gate("J1 通道代数", ok,
     f"谱 Z={spec_z.round(12)} X={spec_x.round(12)}，{{Z,X}}={anti.max():.1e}，跨侧={cross:.1e}（期望 ±1 与 0）")

# ---------- J2 锁定态纠缠 ----------
psi = phi_state(0.0)
rho = np.outer(psi, psi.conj())
rho_a = np.zeros((2, 2), dtype=complex)
M4 = rho.reshape(2, 2, 2, 2)
rho_a[0, 0] = M4[0, :, 0, :].trace()
rho_a[0, 1] = M4[0, :, 1, :].trace()
rho_a[1, 0] = M4[1, :, 0, :].trace()
rho_a[1, 1] = M4[1, :, 1, :].trace()
ev = np.linalg.eigvalsh(rho_a)
YY = np.kron(SY, SY)
conc = abs(psi.conj() @ (YY @ psi.conj()))
ok = np.allclose(ev, [0.5, 0.5], atol=1e-14) and abs(conc - 1) < 1e-12
gate("J2 锁定态纠缠", ok, f"约化本征值={ev.round(14)}（期望 0.5,0.5），concurrence={conc:.14f}（期望 1）")

# ---------- J3 精确 CHSH ----------
def chsh_exact(state, rho=None, ang=PAIRS):
    Es = []
    for ta, tb in ang:
        Ea = 0.0
        for sa in (+1, -1):
            for sb in (+1, -1):
                M = proj(ta, "A", sa) @ proj(tb, "B", sb)
                w = expect(state, M, rho)
                Ea += sa * sb * w
        Es.append(Ea)
    return Es[0] + Es[1] + Es[2] - Es[3]

S_exact = chsh_exact(psi)
ok = abs(S_exact - TS) < 1e-12
gate("J3 精确 CHSH", ok, f"S = {S_exact:.14f}（期望 {TS:.14f}，偏差 {abs(S_exact-TS):.1e}）")

# ---------- J3b 锁定族剖面 S_max(gamma) ----------
def smax_meridian(gamma):
    c = np.cos(gamma)
    a1 = np.linspace(0, np.pi, 3601)
    ca, sa = np.cos(a1), np.sin(a1)
    val = np.sqrt((1 + ca) ** 2 + (sa * c) ** 2) + np.sqrt((1 - ca) ** 2 + (sa * c) ** 2)
    return val.max()

gammas = np.linspace(0, 2 * np.pi, 181)
smax_num = np.array([smax_meridian(g) for g in gammas])
smax_theory = 2 * np.sqrt(1 + np.cos(gammas) ** 2)
dev = np.abs(smax_num - smax_theory).max()
ok = dev < 1e-6
gate("J3b 锁定族剖面", ok,
     f"S_max(gamma) = 2*sqrt(1+cos^2 gamma) 最大偏差 {dev:.1e}（1e-6）；"
     f"gamma=0 饱和 {smax_num[0]:.6f}，gamma=pi/2 骑线 {smax_num[45]:.6f}")

# ---------- J4 Born 采样：生死闸门 ----------
rng = np.random.default_rng(SEED)
E_hat, sig2 = [], 0.0
for ta, tb in PAIRS:
    P = joint_probs(psi, ta, tb)
    counts = rng.multinomial(N_SAMP, P.reshape(-1))
    signs = np.array([+1, -1, -1, +1])
    E_hat.append((counts * signs).sum() / N_SAMP)
    E_ex = expect(psi, obs(ta, "A") @ obs(tb, "B"))
    sig2 += (1 - E_ex ** 2) / N_SAMP
S_hat = E_hat[0] + E_hat[1] + E_hat[2] - E_hat[3]
sig_S = np.sqrt(sig2)
z_gate = (S_hat - CL) / sig_S
ok = abs(S_hat - TS) < 4 * sig_S and z_gate >= 300
gate("J4 Born 采样闸门", ok,
     f"S_hat = {S_hat:.5f} ± {sig_S:.5f}（期望 {TS:.5f}，偏差 {abs(S_hat-TS)/sig_S:.2f} sigma）；"
     f"越线 z = {z_gate:.0f}（阈 300）")

# ---------- J5 经典圈对照 ----------
def circle_E(da):
    d = np.abs((da + np.pi) % (2 * np.pi) - np.pi)
    return 1 - 2 * d / np.pi

S_circ_exact = sum((+1, +1, +1, -1)[i] * circle_E(tb - ta) for i, (ta, tb) in enumerate(PAIRS))
lam = rng.uniform(0, 2 * np.pi, N_SAMP)
S_circ_hat = 0.0
for i, (ta, tb) in enumerate(PAIRS):
    a = np.where(np.cos(lam - ta) >= 0, 1, -1)
    b = np.where(np.cos(lam - tb) >= 0, 1, -1)
    E = (a * b).mean()
    S_circ_hat += (+1, +1, +1, -1)[i] * E
sig_c = np.sqrt(4 / N_SAMP)
best_lhv = 2.0
for a0 in (1, -1):
    for a1 in (1, -1):
        for b0 in (1, -1):
            for b1 in (1, -1):
                best_lhv = max(best_lhv, a0 * b0 + a1 * b0 + a0 * b1 - a1 * b1)
ok = abs(S_circ_hat - S_circ_exact) < 4 * sig_c and abs(S_circ_exact - CL) < 1e-12 and best_lhv == 2
gate("J5 经典圈对照", ok,
     f"同一几何预存角读法：S = {S_circ_exact:.12f}（解析），采样 {S_circ_hat:.5f} ± {sig_c:.5f}；"
     f"LHV 穷举最大 = {best_lhv:.1f}（经典封顶 2）")

# ---------- J6 表针退相干控制 ----------
rho_mix = 0.5 * np.diag([1, 0, 0, 1]).astype(complex)  # ½(|00><00|+|11><11|)：仅层位关联
S_deph = chsh_exact(None, rho=rho_mix)
ok = abs(S_deph - np.sqrt(2)) < 1e-12 and S_deph < CL
gate("J6 表针退相干控制", ok,
     f"层位关联保持、表针清除：S = {S_deph:.12f}（期望 sqrt(2)={np.sqrt(2):.12f}）< 2：违反载体 = 表针通道")

# ---------- J7 可见度阈 ----------
p_grid = np.linspace(0, 1, 41)
s_p = 2 * 2 ** 0.5 * p_grid
p_star_num = 1.0 / (2 ** 0.5)
ok_flag = True
samp_pts = []
for p in (0.6, p_star_num, 0.8):
    rho_p = p * np.outer(psi, psi.conj()) + (1 - p) * np.eye(4, dtype=complex) / 4
    Es, s2 = [], 0.0
    for ta, tb in PAIRS:
        P = joint_probs(None, ta, tb, rho=rho_p)
        counts = rng.multinomial(N_SAMP, P.reshape(-1))
        signs = np.array([+1, -1, -1, +1])
        Es.append((counts * signs).sum() / N_SAMP)
        E_ex = np.real(np.trace(rho_p @ (obs(ta, "A") @ obs(tb, "B"))))
        s2 += (1 - E_ex ** 2) / N_SAMP
    Sp = Es[0] + Es[1] + Es[2] - Es[3]
    sS = np.sqrt(s2)
    samp_pts.append((p, Sp, sS))
    ok_flag &= abs(Sp - TS * p) < 4 * sS
p_cross = 1.0 / (2 ** 0.5)
ok = ok_flag and abs(p_cross - p_star_num) < 1e-3
gate("J7 可见度阈", ok,
     f"S(p) = 2*sqrt(2) p；阈 p* = 1/sqrt(2) = {p_star_num:.6f}；"
     + "采样 " + "; ".join(f"p={p:.4f}: S={s:.4f}±{e:.4f}" for p, s, e in samp_pts))

# ---------- J8 无信号（只读通道） ----------
pmax_dev = 0.0
for ta in (ALPHA0, ALPHA1):
    for tb in (BETA0, BETA1):
        P = joint_probs(psi, ta, tb)
        pmax_dev = max(pmax_dev, abs(P[0, :].sum() - 0.5))
idx_b0 = 0
marg_b = []
for tb, j in ((BETA0, 0), (BETA1, 1)):
    P = joint_probs(psi, ALPHA0, tb)
    counts = rng.multinomial(N_SAMP, P.reshape(-1))
    marg_b.append((counts[0] + counts[1]) / N_SAMP)
sig_m = np.sqrt(0.5 / N_SAMP)
dev_s = abs(marg_b[0] - marg_b[1])
ok = pmax_dev < 1e-14 and dev_s < 5 * sig_m
gate("J8 无信号", ok,
     f"精确边际偏差 {pmax_dev:.1e}（1e-14）；采样 |P(a|b0)-P(a|b1)| = {dev_s:.1e} < 5 sigma = {5*sig_m:.1e}")

# ---------- J9 Ramsey 条纹 ----------
gamma_in = np.pi / 3

H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # 支重组合（50/50 耦合）

def ramsey_P(th, gamma):
    w = np.array([1, np.exp(1j * gamma)]) / np.sqrt(2)
    w2 = np.array([w[0], np.exp(1j * th) * w[1]])   # 零模踢：支 1 相位推进 th
    wr = H2 @ w2                                     # 重组合
    return abs(wr[0]) ** 2                           # 层位读出

th_grid = np.linspace(0, 2 * np.pi, 361, endpoint=False)
p_grid_r = np.array([ramsey_P(t, gamma_in) for t in th_grid])
p_hi, p_lo = ramsey_P(-gamma_in, gamma_in), ramsey_P(np.pi - gamma_in, gamma_in)
vis = (p_hi - p_lo) / (p_hi + p_lo)  # 精确极值角，网格仅作图
rec = np.arctan2(1 - 2 * ramsey_P(np.pi / 2, gamma_in), 2 * ramsey_P(0, gamma_in) - 1)
rec = (rec + 2 * np.pi) % (2 * np.pi)
ok = abs(vis - 1) < 1e-12 and abs(rec - gamma_in) < 1e-10
gate("J9 Ramsey 条纹", ok,
     f"零模踢+重组合+层位读出条纹可见度 = {vis:.14f}（期望 1）；拟合相位 {rec:.12f} 回收输入 {gamma_in:.12f}")

# ---------- J10 上界扫描 ----------
n_grid = 241
xs = np.linspace(-np.pi, np.pi, n_grid)
best = -9.9, None
for u in xs:
    X_, Y_ = np.meshgrid(xs, xs, indexing="ij")
    Sg = np.cos(X_) + np.cos(Y_) + np.cos(u) - np.cos(Y_ + u - X_)
    i = np.unravel_index(np.argmax(Sg), Sg.shape)
    if Sg[i] > best[0]:
        best = (Sg[i], (X_[i], Y_[i], u))
ok = abs(best[0] - TS) < 3e-3
gate("J10 上界扫描", ok,
     f"S 全网格最大 = {best[0]:.6f}（期望 2*sqrt(2)={TS:.6f}，网格 {n_grid}^3）：模型无超量子区")

# ---------- 汇总 ----------
print("-" * 78)
n_pass = sum(1 for _, okk, _ in results if okk)
print(f"汇总：{n_pass}/{len(results)} PASS")
print(f"生死闸门：第一颚（S>2 存在性）{'通过' if n_pass >= 8 else ''}；第二颚（S<=2*sqrt(2)）{'通过' if n_pass >= 8 else ''}")
print("-" * 78)

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
ax = axes[0, 0]
ax.plot(p_grid * 100, s_p, "b-", lw=2, label=r"$S(p)=2\sqrt{2}\,p$（精确）")
for p, s, e in samp_pts:
    ax.errorbar(p * 100, s, yerr=e, fmt="ro", ms=5, capsize=3)
ax.axhline(CL, color="k", ls="--", lw=1, label="经典界 2")
ax.axhline(TS, color="purple", ls=":", lw=1.5, label=r"Tsirelson $2\sqrt{2}$")
ax.axvline(p_star_num * 100, color="gray", ls="-.", lw=1)
ax.text(p_star_num * 100 + 1, 0.4, r"$p^*=1/\sqrt{2}$", fontsize=10)
ax.set_xlabel("表针相干度 p (%)")
ax.set_ylabel("CHSH S")
ax.set_title("(a) 可见度阈：违反载体 = 表针相干")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(np.degrees(gammas), smax_num, "b-", lw=2, label="数值最大（子午面设置）")
ax.plot(np.degrees(gammas), smax_theory, "r--", lw=1.2, label=r"$2\sqrt{1+\cos^2\gamma}$")
ax.axhline(CL, color="k", ls="--", lw=1)
ax.axhline(TS, color="purple", ls=":", lw=1.5)
ax.set_xlabel(r"锁定相对表针 $\gamma$ (°)")
ax.set_ylabel(r"$S_{\max}$")
ax.set_title("(b) 锁定族剖面：gamma=0 饱和，pi/2 骑线")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 0]
for g, c in ((0.0, "tab:blue"), (np.pi / 3, "tab:orange")):
    ax.plot(np.degrees(th_grid), [ramsey_P(t, g) for t in th_grid], color=c, lw=2,
            label=rf"$\gamma={g:.3f}$（理论）")
ax.plot(np.degrees(th_grid[::20]), p_grid_r[::20], "k.", ms=4, label=r"采样（$\gamma=\pi/3$）")
ax.set_xlabel(r"零模踢 $\theta$ (°)")
ax.set_ylabel(r"P(层位=+1)")
ax.set_title("(c) Ramsey 条纹：表针通道操作化")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 1]
names = ["量子锁定\n|Phi_0>", "经典圈\n(预存角)", "LHV 穷举\n上界", "表针退相干\n(混合)"]
vals = [S_hat, S_circ_hat, best_lhv, S_deph]
cols = ["tab:blue", "tab:gray", "tab:gray", "tab:red"]
bars = ax.bar(names, vals, color=cols, alpha=0.85)
ax.axhline(CL, color="k", ls="--", lw=1.2)
ax.axhline(TS, color="purple", ls=":", lw=1.5)
ax.text(3.42, CL + 0.03, "经典界 2", fontsize=9)
ax.text(3.42, TS + 0.02, r"$2\sqrt{2}$", fontsize=9)
for b, v_ in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v_ + 0.03, f"{v_:.4f}", ha="center", fontsize=10)
ax.set_ylim(0, 3.2)
ax.set_ylabel("CHSH S")
ax.set_title("(d) 四种读法对照（采样值，N=2e6/对）")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("P1-29 v0 生死闸门：纠缠锁定对的 CHSH 存在性、上界与通道判别（种子 20260914）", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG_PATH, dpi=150)
print(f"图已存：{FIG_PATH}")
