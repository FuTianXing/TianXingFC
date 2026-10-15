# -*- coding: utf-8 -*-
"""判据第 644 号：DOGA 级联的阈值结构与界域缺口（642 跟进轮）

背景：判据第 642 号确认原点稳定性由 r=alpha/beta 与 1 的比较决定。
本轮跟进两问：其一，体系要"非零序参量"（兴）对 r 有何要求；其二，附件 D 设计原则 1
称"Logistic 自限项保证变量在 (0,1] 内演化"是否成立。

理论期望（先立期望后实现）：
  G1 双阈值结构：D 方程在 O=G=0、A->1 时的非零平衡为 D_inf = 1 - 1/r_D，
     故 r_D > 1 是 D 非零的必要条件；O、G 的线性率 alpha_X - beta_X*A 在 A->1 时为
     beta_X(r_X - 1)，故 r_X > 1 是其增长的条件。结构对称性（附件 D 原则 2：D,O,G
     地位等价）给 r_D = r_O = r_G = r。
  G2 越界（界域缺口）：交叉项 gamma*Y*Z 随参量增长无上界，logistic 项在 X>1 后
     变负但仍为 -alpha*X*(X-1)（量级 X^2），而交叉项量级为 gamma*X^2（两参量同增时）。
     故 gamma 够大时 logistic 压制不住，X 越界。附件 D 原则 1 的"logistic 保证界域"
     只在无交叉项或 gamma 小时成立。
  G3 三档行为：r<1 全部衰减到零（原点稳定，无种子则衰减、有种子被压回）；
     r=1 临界（一阶为零，由二阶与交叉项给出缓慢漂移）；r>1 原点不稳定（快速越界）。
  G4 恒真反例：加界域饱和项（X>1 时钳制）后越界消失，证明越界来自界域约束缺失
     而非数值方法。

判据设计纪律：容差从误差预算推导；恒真反例必配；多档扫描；协议伪影分层报告。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images"


def doga(t, y, p):
    D, O, G, A = y
    f = p["seed"] * np.exp(-((t - p["u0"]) / p["w"]) ** 2)
    dD = p["aD"] * D * (1 - D) - p["bD"] * A * D + p["gD"] * O * G + f
    dO = p["aO"] * O * (1 - O) - p["bO"] * A * O + p["gO"] * G * D
    dG = p["aG"] * G * (1 - G) - p["bG"] * A * G + p["gG"] * O * D
    dA = p["aA"] * A * (1 - A) + p["bA"] * (1 - D) * O * G - p["dl"] * A * D
    return [dD, dO, dG, dA]


def doga_clamped(t, y, p):
    """加界域饱和项：正向三元组 X>1 时以强反馈钳制（软壁，k 足够大以压住交叉项）。"""
    dy = doga(t, y, p)
    k = p.get("kappa_clamp", 1e4)
    for i in range(3):
        if y[i] > 1.0:
            dy[i] -= k * (y[i] - 1.0)
        if y[i] < 0.0:
            dy[i] -= k * y[i]
    return dy


def base(r=1.0, g=0.5, seed=0.3):
    return dict(aD=r, bD=1.0, gD=g, aO=r, bO=1.0, gO=g, aG=r, bG=1.0, gG=g,
                aA=1.0, bA=0.3, dl=1.0, seed=seed, u0=5.0, w=1.5, kappa_clamp=1e4)


def integ(p, y0, T, n=20001, use_clamp=False):
    fn = doga_clamped if use_clamp else doga
    dt = T / (n - 1)
    Y = np.zeros((n, 4))
    Y[0] = y0
    for i in range(1, n):
        y = Y[i - 1]
        k1 = np.array(fn(0, y, p))
        k2 = np.array(fn(0, y + 0.5 * dt * k1, p))
        k3 = np.array(fn(0, y + 0.5 * dt * k2, p))
        k4 = np.array(fn(0, y + dt * k3, p))
        yn = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(yn)) or np.max(np.abs(yn)) > 1e12:
            Y[i:] = np.nan
            return np.linspace(0, T, n), Y, True
        Y[i] = yn
    return np.linspace(0, T, n), Y, False


T = 200.0

# ---------------------------------------------------------------- 冒烟
print("=" * 78)
print("冒烟检查")
print("=" * 78)
p = base(r=1.1)
y1 = np.array([1.0, 1.0, 1.0, 1.0])
F = np.array(doga(0, y1, p))
pred_O = p["gO"] * 1.0 * 1.0 - p["bO"] * 1.0   # X=1 处 logistic 项为零
print("  y=(1,1,1,1)：dO/du 实测 %.6f，解析 gamma*G*D - beta*A = %.6f" % (F[1], pred_O))
assert abs(F[1] - pred_O) < 1e-12, "冒烟失败"
print("  -> X=1 处 logistic 项为零，方向由交叉项 gamma*Y*Z 与抑制项 beta*A 之差决定")
print("  冒烟通过。")

# ---------------------------------------------------------------- G1
print()
print("=" * 78)
print("G1 双阈值结构：D 非零需 r_D>1；O,G 增长需 r_X>1")
print("=" * 78)
print("  G1a D 的非零平衡 D_inf = 1 - 1/r_D（O=G=0、A=1 处）：")
G1a_rows = []
for r in [0.5, 0.9, 1.0]:
    p = base(r=r, seed=0.0)
    t, Y, blown = integ(p, [1e-4, 1e-6, 1e-6, 0.9], T=T)
    D_end = np.nan if blown else Y[-1, 0]
    ok = (not blown) and D_end < 1e-3
    G1a_rows.append((r, D_end, ok))
    print("    r=%.1f: D末=%.3e -> %s %s" % (r, D_end,
          "衰减到零（r<=1，无方向性）" if ok else "非零", "✓" if ok else "✗"))
G1a = all(x[2] for x in G1a_rows)
print("    （r>1 档 D 趋向 D_inf>0，但该平衡被 O,G 增长破坏，见 G2/G3）")

print("  G1b O,G 的线性率在冻结 A=0.9 时为 lambda = r - 0.9：")
G1b_rows = []
for r in [0.5, 0.9, 1.1]:          # 0.5 给负、0.9 给零（恰在阈值）、1.1 给正
    p = base(r=r)
    p["aA"] = 0.0; p["bA"] = 0.0; p["dl"] = 0.0   # 冻结 A=0.9
    p["aD"] = 0.0; p["bD"] = 0.0; p["gD"] = 0.0   # 冻结 D=0（隔离 O 通道）
    eps = 1e-9
    t, Y, _ = integ(p, [0.0, eps, eps, 0.9], T=0.02, n=2001)
    lam = np.log(Y[-1, 1] / eps) / 0.02
    lam_ana = r - 0.9
    ok = abs(lam - lam_ana) < 1e-3
    G1b_rows.append((r, lam, ok))
    print("    r=%.1f: 实测 lambda=%+.6f（解析 %+.6f）-> %s %s"
          % (r, lam, lam_ana,
             "增长" if lam > 1e-9 else ("衰减" if lam < -1e-9 else "临界（恰在阈值）"),
             "✓" if ok else "✗"))
G1b = all(x[2] for x in G1b_rows)
G1 = G1a and G1b
print("  G1 %s：双阈值结构成立（r_D>1 给 D 非零、r_X>1 给 O,G 增长；结构对称性给三者同 r）"
      % ("PASS" if G1 else "FAIL"))

# ---------------------------------------------------------------- G2
print()
print("=" * 78)
print("G2 越界机制：交叉项无上界，logistic 压制不足")
print("=" * 78)
print("  G2a X=1 处的导数符号（A=1、其余正向参量=1）随 gamma：")
G2a_rows = []
for g in [0.5, 1.0, 1.1, 2.0]:
    dX = g * 1.0 * 1.0 - 1.0 * 1.0
    over = dX > 0
    exp_over = (g > 1.0)
    ok = (over == exp_over)
    G2a_rows.append((g, dX, ok))
    print("    gamma=%.1f: dX/du|_(X=1) = %+.3f -> %s %s"
          % (g, dX, "交叉项压过抑制（越界）" if over else "抑制压过交叉项（界内）",
             "✓" if ok else "✗"))
G2a = all(x[2] for x in G2a_rows)
print("    -> 阈值 gamma_c = beta*A = 1（A=1 时）：gamma>1 即越界，与 r 无关")

print("  G2b 完整积分下的越界（gamma=0.5 < gamma_c，但 r>1 档仍越界）：")
G2b_rows = []
for r in [0.9, 1.0, 1.1, 1.5]:
    p = base(r=r)
    t, Y, blown = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T)
    fin = np.isfinite(Y[:, 1])
    Omax = np.nanmax(Y[fin, 1]) if fin.any() else np.nan
    over = blown or Omax > 1.0
    exp_over = (r >= 1.1)
    ok = (over == exp_over)
    G2b_rows.append((r, Omax, ok))
    print("    r=%.1f: max O = %.4e -> %s %s"
          % (r, Omax, "越界" if over else "界内", "✓" if ok else "✗"))
G2b = all(x[2] for x in G2b_rows)
print("    -> r>1 档 O,G 增长后交叉项被放大（量级 gamma*X^2），最终压过 logistic")
G2 = G2a and G2b
print("  G2 %s：越界由交叉项无上界引起，附件 D 原则 1 的\"logistic 保证界域\"不成立"
      % ("PASS" if G2 else "FAIL"))

# ---------------------------------------------------------------- G3
print()
print("=" * 78)
print("G3 三档行为：r<1 衰减、r=1 临界缓慢漂移、r>1 快速越界")
print("=" * 78)
G3_rows = []
for r, lab in [(0.9, "衰减"), (1.0, "临界"), (1.1, "越界")]:
    p = base(r=r)
    t, Y, blown = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T)
    fin = np.isfinite(Y[:, 1])
    Omax = np.nanmax(Y[fin, 1]) if fin.any() else np.nan
    Oend = np.nan if blown else Y[-1, 1]
    if r < 1:
        ok = (not blown) and Oend < 1e-6
        desc = "衰减到零"
    elif r == 1:
        ok = (not blown) and (1e-4 < Oend < 1.0)
        desc = "缓慢漂移（量级 %.2e）" % Oend
    else:
        ok = blown or Omax > 1.0
        desc = "越界（max %.2e）" % Omax
    G3_rows.append((r, Oend, Omax, ok, desc))
    print("    r=%.1f: %s %s" % (r, desc, "✓" if ok else "✗"))
G3 = all(x[3] for x in G3_rows)
print("  G3 %s：三档行为分明" % ("PASS" if G3 else "FAIL"))

print("  G3b r=1 档的时间尺度（长窗）：")
p = base(r=1.0)
for Tlong in [200.0, 1000.0, 5000.0]:
    t, Y, blown = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=Tlong, n=int(Tlong * 100) + 1)
    fin = np.isfinite(Y[:, 1])
    Omax = np.nanmax(Y[fin, 1]) if fin.any() else np.nan
    print("    T=%.0f: %s（max O=%.3e）"
          % (Tlong, "越界" if blown else "界内", Omax))

# ---------------------------------------------------------------- G4
print()
print("=" * 78)
print("G4 恒真反例：加界域饱和项后越界消失")
print("=" * 78)
G4_rows = []
for r in [1.1, 1.5]:
    p = base(r=r)
    t1, Y1, blown1 = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T, use_clamp=False)
    t2, Y2, blown2 = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T, use_clamp=True)
    over_free = blown1 or (np.nanmax(Y1[:, 1]) > 1.0)
    over_clamp = blown2 or (np.nanmax(Y2[:, 1]) > 1.0 + 1e-3)
    ok = over_free and (not over_clamp)
    G4_rows.append((r, ok))
    print("    r=%.1f: 无界域约束 max O=%.3e（%s）| 加饱和项 max O=%.6f（%s）%s"
          % (r, np.nanmax(Y1[:, 1]), "越界" if over_free else "界内",
             np.nanmax(Y2[:, 1]), "越界" if over_clamp else "界内", "✓" if ok else "✗"))
G4 = all(x[1] for x in G4_rows)
print("  G4 %s：越界来自界域约束缺失（加约束即消失），非数值方法" % ("PASS" if G4 else "FAIL"))

# ---------------------------------------------------------------- 出图
print()
print("=" * 78)
print("出图")
print("=" * 78)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# 面板 1：三档 O 轨迹
ax = axes[0]
for r, c, lab in [(0.9, "tab:blue", r"$r=0.9<1$（衰减）"),
                  (1.0, "tab:orange", r"$r=1$（临界漂移）"),
                  (1.1, "tab:green", r"$r=1.1>1$（越界）")]:
    p = base(r=r)
    t, Y, _ = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T)
    ax.plot(t, Y[:, 1], color=c, lw=1.6, label=lab)
ax.axhline(1.0, color="k", ls=":", lw=1, label="归一化上界 1")
ax.set_yscale("log")
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel(r"$\tilde O$")
ax.set_title("三档行为（G3）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 面板 2：gamma 阈值（G2a）
ax = axes[1]
gs = np.linspace(0.2, 2.0, 200)
dX = gs - 1.0
ax.plot(gs, dX, "k-", lw=1.6)
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(1.0, color="tab:red", ls="--", lw=1.2, label=r"$\gamma_c=\beta A=1$")
ax.fill_between(gs, 0, dX, where=(dX > 0), color="tab:red", alpha=0.2, label="越界区")
ax.fill_between(gs, 0, dX, where=(dX < 0), color="tab:blue", alpha=0.2, label="界内区")
ax.set_xlabel(r"$\gamma$（交叉催化系数）")
ax.set_ylabel(r"$dX/du\vert_{X=1}$")
ax.set_title("界域缺口：$X=1$ 处导数符号（G2a）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 面板 3：恒真反例
ax = axes[2]
p = base(r=1.1)
t1, Y1, _ = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T, use_clamp=False)
t2, Y2, _ = integ(p, [0.0, 1e-3, 1e-3, 0.9], T=T, use_clamp=True)
ax.plot(t1, Y1[:, 1], color="tab:green", lw=1.6, label="无界域约束（越界）")
ax.plot(t2, Y2[:, 1], color="tab:red", lw=1.6, label="加饱和项（钳在 1）")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_yscale("log")
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel(r"$\tilde O$")
ax.set_title(r"恒真反例：界域约束消除越界（G4，$r=1.1$）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
out = IMG + r"\doga_boundary_gap_644.png"
fig.savefig(out, dpi=150)
print("  图已存：", out)

# ---------------------------------------------------------------- 汇总
print()
print("=" * 78)
print("判据汇总")
print("=" * 78)
for name, val, desc in [("G1", G1, "双阈值结构：D 非零需 r_D>1、O,G 增长需 r_X>1（结构对称性给同 r）"),
                        ("G2", G2, "越界由交叉项无上界引起，logistic 不保界域（gamma_c=beta*A）"),
                        ("G3", G3, "三档分明：r<1 衰减、r=1 临界漂移、r>1 越界"),
                        ("G4", G4, "加界域约束即消除越界（恒真反例）")]:
    print("  %s %s  %s" % (name, "PASS" if val else "FAIL", desc))
print()
print("  结论：体系要非零序参量（兴）须 r>1（G1），但 r>1 档在无界域约束下必越界（G2/G3）。")
print("        故真缺口是界域约束，与 7.9 发现二、三及判据第 642 号 E5 同源。")
print("        附件 D 设计原则 1 的\"logistic 自限项保证界域\"不成立（交叉项破坏）。")
