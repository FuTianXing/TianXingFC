# -*- coding: utf-8 -*-
"""判据第 642 号：DOGA 原点稳定性复核与级联阈值条件化（P4-26）

背景：库内诊断（注记-温度定义 §7.9 发现一，登记 P4-26）称原点 (D,O,G)=(0,0,0)
"各项在原点消失、无线性增长通道"，故 A 下降后 O、G 仍钉在原点，级联不成立。
本脚本复核该诊断，给出正确判据。

理论期望（先立期望后实现）：
  E1 不变集：O=G=0 精确初值下 O、G 全程机器零。这是原诊断"钉在原点"的读数来源，
     只能说明精确零是解（不变集），不能说明邻域无增长通道。
  E2 线性化通道：dO/du = aO*O(1-O) - bO*A*O + gO*G*D，原点对 O 的偏导
     lambda_O = aO - bO*A。该通道明确存在、一般非零，故"无线性增长通道"为错。
  E3 渐近阈值：A 方程在 D=O=G=0 处为 dA/du = aA*A(1-A)，稳定不动点 A_inf = 1
     （aA>0）。故 O、G 增长之前 A 已被推向 1，阈值不是 A0 而是
        lambda_O -> aO - bO*1 = bO*(r-1),   r := aX/bX
     即阈值 r_c = 1，与 A0 无关。r<1 原点渐近稳定（O 衰减）；r>1 不稳定（级联自发）；
     r=1 临界慢化（一阶项消失，需二阶）。
  E4 恒真反例：把判据写成"精确零初值下 O 保持零"，则对任何 r 都 PASS（恒真、无信息）；
     换成非零微扰后，正确判据在 r<1 与 r>1 两侧给出相反结果（判据翻，有区分力）。
  E5 界域伪影：方程不保 [0,1]，r>1 档下 O 越界、A 转负，故自发"级联"读数为发散，
     不作物理读数（与 7.9 发现二、三同源）。

判据设计纪律：容差从误差预算推导；恒真反例必配；多档扫描；协议伪影分层报告。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images"

# ----------------------------------------------------------------------------
# 模型：附件 D 完整 DOGA 方程组
# ----------------------------------------------------------------------------
def doga(t, y, p):
    D, O, G, A = y
    f = p["seed"] * np.exp(-((t - p["u0"]) / p["w"]) ** 2)
    dD = p["aD"] * D * (1 - D) - p["bD"] * A * D + p["gD"] * O * G + f
    dO = p["aO"] * O * (1 - O) - p["bO"] * A * O + p["gO"] * G * D
    dG = p["aG"] * G * (1 - G) - p["bG"] * A * G + p["gG"] * O * D
    dA = p["aA"] * A * (1 - A) + p["bA"] * (1 - D) * O * G - p["dl"] * A * D
    return [dD, dO, dG, dA]


def base_params(aX=1.0, bX=1.0, **over):
    p = dict(aD=aX, bD=bX, gD=0.5,
             aO=aX, bO=bX, gO=0.5,
             aG=aX, bG=bX, gG=0.5,
             aA=1.0, bA=0.3, dl=1.0,
             seed=0.3, u0=5.0, w=1.5)
    p.update(over)
    return p


def integrate(p, y0, T, n=20001):
    """定步长 RK4；越界（|y|>1e12 或非有限）即截断并标 blown。"""
    dt = T / (n - 1)
    Y = np.zeros((n, 4))
    Y[0] = y0
    for i in range(1, n):
        y = Y[i - 1]
        k1 = np.array(doga(0, y, p))
        k2 = np.array(doga(0, y + 0.5 * dt * k1, p))
        k3 = np.array(doga(0, y + 0.5 * dt * k2, p))
        k4 = np.array(doga(0, y + dt * k3, p))
        yn = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(yn)) or np.max(np.abs(yn)) > 1e12:
            Y[i:] = np.nan
            return np.linspace(0, T, n), Y, True
        Y[i] = yn
    return np.linspace(0, T, n), Y, False


T_LONG = 200.0

# ----------------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------------
print("=" * 78)
print("冒烟检查")
print("=" * 78)
p = base_params(aX=1.0, bX=1.0)
y0 = np.array([0.0, 1e-6, 1e-6, 0.9])
F = np.array(doga(0, y0, p))
print("  y0 = [D,O,G,A] =", y0)
print("  逐轴相对变化率：O %.6e  G %.6e" % (abs(F[1]) / y0[1], abs(F[2]) / y0[2]))
# 容差预算：相对率 = aO - bO*A - aO*O，末项即 logistic 修正，量级 O(1e-6)
tol_smoke = 2 * y0[1]
assert abs(abs(F[1]) / y0[1] - (1.0 - 1.0 * 0.9)) < tol_smoke, "冒烟：线性率不符"
print("  -> 变化轴 = 被微扰轴 O,G；相对率 = aO - bO*A（差为 logistic 截断 %.1e）" % y0[1])
print("  冒烟通过。")

# ----------------------------------------------------------------------------
# E1 不变集
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("E1 不变集：O=G=0 精确初值")
print("=" * 78)
t, Y, _ = integrate(base_params(), [0.0, 0.0, 0.0, 0.9], T=T_LONG)
E1 = (Y[-1, 1] == 0.0) and (Y[-1, 2] == 0.0)
print("  末态 O=%.3e G=%.3e" % (Y[-1, 1], Y[-1, 2]))
print("  E1 %s：精确零是不变集（原诊断读数来源）" % ("PASS" if E1 else "FAIL"))

# ----------------------------------------------------------------------------
# E2 线性化通道（冻结 A 隔离）
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("E2 线性化通道：冻结 A，短时增长率 lambda_O = aO - bO*A")
print("=" * 78)
eps = 1e-9
E2_rows = []
for aX, bX, A0 in [(1.0, 1.0, 0.9), (1.5, 1.0, 0.9), (0.5, 1.0, 0.9),
                   (1.0, 1.0, 0.99), (2.0, 1.0, 0.5)]:
    p = base_params(aX=aX, bX=bX)
    p["aA"] = 0.0; p["bA"] = 0.0; p["dl"] = 0.0   # 冻结 A
    tw = 0.02
    t, Y, _ = integrate(p, [0.0, eps, eps, A0], T=tw, n=2001)
    lam_num = np.log(Y[-1, 1] / eps) / tw
    lam_ana = aX - bX * A0
    rel = abs(lam_num - lam_ana) / max(abs(lam_ana), 1e-12)
    E2_rows.append((aX, bX, A0, lam_ana, lam_num, rel))
    print("  aO=%.1f bO=%.1f A=%.2f: 解析 %+.6f 实测 %+.6f 相对差 %.2e"
          % (aX, bX, A0, lam_ana, lam_num, rel))
E2 = all(r[5] < 1e-3 for r in E2_rows)
print("  E2 %s：线性通道存在且=解析值（'无线性增长通道'为错）" % ("PASS" if E2 else "FAIL"))

# ----------------------------------------------------------------------------
# E3 渐近阈值：A_inf = 1，阈值 r_c = 1（与 A0 无关）
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("E3 渐近阈值：A_inf = 1（A 方程稳定不动点），故 lambda_O -> bO*(r-1)")
print("=" * 78)
print("  E3a 阈值定位在 r_c = 1：")
ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
E3a_rows = []
for r in ratios:
    p = base_params(aX=r, bX=1.0)
    t, Y, blown = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
    Oend = np.nan if blown else Y[-1, 1]
    if blown:
        cls = "越界（不稳定）"
    elif Oend < 1e-10:
        cls = "衰减到零（渐近稳定）"
    elif Oend < 1e-3:
        cls = "临界慢化（保持量级）"
    else:
        cls = "增长"
    exp = "不稳定" if r > 1 else ("临界" if r == 1 else "稳定")
    if r > 1:
        ok = blown
    elif r < 1:
        ok = (not blown) and Oend < 1e-10
    else:
        ok = (not blown) and (1e-9 < Oend < 1e-3)
    E3a_rows.append((r, exp, cls, ok))
    print("    r=%.1f: 期望%s 实际%s [O末=%s] %s"
          % (r, exp, cls, "nan" if blown else "%.3e" % Oend, "✓" if ok else "✗"))
E3a = all(x[3] for x in E3a_rows)

print("  E3b 阈值与 A0 无关（A 被推向 A_inf=1）：")
E3b_rows = []
for A0 in [0.5, 0.9, 0.99]:
    for r in [0.9, 1.1]:
        p = base_params(aX=r, bX=1.0)
        t, Y, blown = integrate(p, [0.0, 1e-6, 1e-6, A0], T=T_LONG)
        if r > 1:
            ok = blown
        else:
            ok = (not blown) and Y[-1, 1] < 1e-10
        E3b_rows.append((A0, r, ok))
        print("    A0=%.2f r=%.1f: %s %s"
              % (A0, r, "越界" if blown else "O末=%.2e" % Y[-1, 1], "✓" if ok else "✗"))
E3b = all(x[2] for x in E3b_rows)

print("  E3c A_inf 的数值确认（r=0.5 档，A 从 0.9 涨到不动点附近）：")
p = base_params(aX=0.5, bX=1.0)
t, Y, _ = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
A_end = Y[-1, 3]
D_end = Y[-1, 0]
# 误差预算：A 的不动点满足 aA*A(1-A) = dl*A*D，即 1-A_inf = dl*D/aA。
# D 稳态非零（约 4e-5），故 A_inf 略低于 1，偏差 = dl*D_end/aA。
budget = p["dl"] * D_end / p["aA"]
dev = 1.0 - A_end
print("    A 末值 = %.10f（解析 A_inf = 1，偏差由 D 项抑制给：dl*D/aA = %.2e）"
      % (A_end, budget))
print("    实测偏差 1-A = %.2e，与预算同量级；相对初始 1-A0 = %.1e 缩小 %.0f 倍"
      % (dev, 1.0 - 0.9, (1.0 - 0.9) / max(dev, 1e-30)))
E3c = (dev < 10 * budget) and (dev < 0.01)
print("    E3c %s（容差取 10 倍预算，覆盖 D 演化的残余）" % ("PASS" if E3c else "FAIL"))

E3 = E3a and E3b and E3c
print("  E3 %s：阈值 r_c = 1（非 A0=0.9）" % ("PASS" if E3 else "FAIL"))

# ----------------------------------------------------------------------------
# E4 恒真反例
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("E4 恒真反例：错误判据（精确零初值）无区分力")
print("=" * 78)
taut_rows = []
for r in [0.5, 1.0, 2.0]:
    p = base_params(aX=r, bX=1.0)
    # 错误判据：精确零初值下 O 保持零
    _, Yz, _ = integrate(p, [0.0, 0.0, 0.0, 0.9], T=T_LONG)
    wrong_pass = (abs(Yz[-1, 1]) < 1e-12)
    # 正确判据：非零微扰 + 长窗，按 r 与 1 的关系分类
    _, Yp, blown = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
    if r > 1:
        right_verdict = "不稳定"
        right_ok = blown
    elif r < 1:
        right_verdict = "稳定"
        right_ok = (not blown) and Yp[-1, 1] < 1e-10
    else:
        right_verdict = "临界"
        right_ok = (not blown) and (1e-9 < Yp[-1, 1] < 1e-3)
    taut_rows.append((r, wrong_pass, right_ok))
    print("    r=%.1f: 零初值判据 %s（对任何 r 同结果）| 正确判据判%s %s"
          % (r, "PASS" if wrong_pass else "FAIL", right_verdict, "✓" if right_ok else "✗"))
E4 = all(t[1] for t in taut_rows) and all(t[2] for t in taut_rows)
print("  E4 %s：错误判据两侧同 PASS（恒真无信息），正确判据两侧相反（有区分力）"
      % ("PASS" if E4 else "FAIL"))

# ----------------------------------------------------------------------------
# E5 界域伪影
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("E5 界域伪影：r>1 档下 O 越界、A 转负")
print("=" * 78)
p = base_params(aX=1.1, bX=1.0)
t, Y, blown = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
finite = np.isfinite(Y[:, 1])
Omax = np.nanmax(Y[finite, 1]) if finite.any() else np.nan
Amin = np.nanmin(Y[finite, 3]) if finite.any() else np.nan
print("  O 最大值 = %.3e（归一化上界为 1）" % Omax)
print("  A 最小值 = %.3e（归一化下界为 0）" % Amin)
E5 = blown and (Omax > 1.0) and (Amin < 0.0)
print("  E5 %s：方程不保 [0,1]，自发'级联'为发散伪影" % ("PASS" if E5 else "FAIL"))

# ----------------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("出图")
print("=" * 78)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# 面板 1：三档 r 的 O 轨迹
ax = axes[0]
for r, c, lab in [(0.9, "tab:blue", r"$r=0.9<1$（衰减）"),
                  (1.0, "tab:orange", r"$r=1$（临界慢化）"),
                  (1.1, "tab:green", r"$r=1.1>1$（越界）")]:
    p = base_params(aX=r, bX=1.0)
    t, Y, _ = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
    ax.plot(t, Y[:, 1], color=c, lw=1.6, label=lab)
ax.axhline(1.0, color="k", ls=":", lw=1, label="归一化上界")
ax.set_yscale("log")
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel(r"$\tilde O$")
ax.set_title(r"原点稳定性：阈值 $r_c=\alpha/\beta=1$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 面板 2：A 的轨迹（A_inf = 1）
ax = axes[1]
for r, c, lab in [(0.9, "tab:blue", r"$r=0.9$"), (1.0, "tab:orange", r"$r=1$"),
                  (1.1, "tab:green", r"$r=1.1$")]:
    p = base_params(aX=r, bX=1.0)
    t, Y, _ = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
    ax.plot(t, Y[:, 3], color=c, lw=1.6, label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1, label=r"$A_\infty=1$")
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel(r"$\tilde A$")
ax.set_title(r"$\tilde A$ 被推向不动点 $A_\infty=1$（故阈值非 $A_0$）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 面板 3：阈值扫描
ax = axes[2]
xs = [x[0] for x in E3a_rows]
ys = []
for r in xs:
    p = base_params(aX=r, bX=1.0)
    t, Y, blown = integrate(p, [0.0, 1e-6, 1e-6, 0.9], T=T_LONG)
    ys.append(np.nan if blown else max(Y[-1, 1], 1e-30))
ax.semilogy(xs, ys, "o-", color="tab:purple", lw=1.5)
ax.axvline(1.0, color="k", ls="--", lw=1, label=r"阈值 $r_c=1$")
ax.set_xlabel(r"$r=\alpha/\beta$")
ax.set_ylabel(r"末态 $\tilde O$（越界档留空）")
ax.set_title("阈值扫描：$r_c=1$ 处翻转（E3a）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
out = IMG + r"\doga_origin_stability_642.png"
fig.savefig(out, dpi=150)
print("  图已存：", out)

# ----------------------------------------------------------------------------
# 判据汇总
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("判据汇总")
print("=" * 78)
for name, val, desc in [("E1", E1, "精确零为不变集（原诊断读数来源）"),
                        ("E2", E2, "线性通道存在且=解析值（'无通道'为错）"),
                        ("E3", E3, "阈值 r_c=1（非 A0=0.9），与 A0 无关"),
                        ("E4", E4, "错误判据恒真、正确判据两侧相反"),
                        ("E5", E5, "界域不保，r>1 档读数为发散伪影")]:
    print("  %s %s  %s" % (name, "PASS" if val else "FAIL", desc))
print()
print("  结论：库内诊断'原点无线性增长通道、动力学无法自启'不成立。")
print("        正确判据：A 被 A 方程推向 A_inf=1，故 lambda_O -> bO*(r-1)，")
print("        阈值 r_c = aX/bX = 1。r<1 原点渐近稳定（需种子）；r>1 原点不稳定")
print("        （级联自发）；r=1 临界慢化。脚本原参数 r=1 恰在临界档。")
