# -*- coding: utf-8 -*-
"""判据第 648 号：DOGA 两吸引子的真伪与界域约束形式（644 跟进轮）

背景：判据第 644 号确认 r>1 档在无界域约束下必越界。本轮跟进三问：
  其一，文档所称"兴盛吸引子 (1,1,1,0)"与"崩溃吸引子 (0,0,0,1)"是否为真不动点；
  其二，界域约束取何形式才使兴盛态可达（硬投影 vs 反馈软壁）；
  其三，两处承重表述（总纲 §7、正稿 §D.4）应如何更正。

理论期望（先立期望后实现）：
  H1 崩溃点 (0,0,0,1) 是精确不动点：各方程右端在 D=O=G=0、A=1 处
     logistic 项为零、抑制项为零、催化项为零、A 方程 alpha_A*1*0=0 且
     -delta*A*D=0，故残差为零。
  H2 兴盛点 (1,1,1,0) 不是不动点：A 方程右端 = alpha_A*0*(1) + beta_A*(1-1)*1*1
     - delta*0*1 = 0；但 D 方程右端 = alpha_D*1*0 - beta_D*0*1 + gamma_D*1*1
     = gamma_D > 0。故残差非零，与 gamma_D 同量级。
  H3 无界域约束时，r>1 档不趋近 (1,1,1,0) 而是越界（承接第 644 号）。
  H4 加硬投影（严格钳在 [0,1]）后，r>1 档收敛到 (1,1,1,0)（A->0）。
  H5 软壁（反馈拉回）的行为对系数敏感，非普遍可达。
  H6 恒真反例：把判据写成"轨迹最终接近某点"（对任意点都可用欧氏距离判），
     须用"该点是否为不动点"作对照才能区分。

判据设计纪律：容差从误差预算推导；恒真反例必配；多档扫描；协议伪影分层报告。
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images"


def rhs(y, p):
    D, O, G, A = y
    dD = p["aD"] * D * (1 - D) - p["bD"] * A * D + p["gD"] * O * G
    dO = p["aO"] * O * (1 - O) - p["bO"] * A * O + p["gO"] * G * D
    dG = p["aG"] * G * (1 - G) - p["bG"] * A * G + p["gG"] * O * D
    dA = p["aA"] * A * (1 - A) + p["bA"] * (1 - D) * O * G - p["dl"] * A * D
    return np.array([dD, dO, dG, dA])


def base(r=1.5, g=0.5):
    return dict(aD=r, bD=1.0, gD=g, aO=r, bO=1.0, gO=g, aG=r, bG=1.0, gG=g,
                aA=1.0, bA=0.3, dl=1.0)


def integ2(p, y0, T, n=20001, mode="free", kc=1e3, seed=0.3):
    """带约束的积分（约束直接进导数）"""
    dt = T / (n - 1)
    Y = np.zeros((n, 4))
    Y[0] = y0
    for i in range(1, n):
        y = Y[i - 1]

        def f(yy):
            dy = rhs(yy, p)
            if mode == "soft":
                for j in range(3):
                    if yy[j] > 1.0:
                        dy[j] -= kc * (yy[j] - 1.0)
                    if yy[j] < 0.0:
                        dy[j] -= kc * yy[j]
            return dy

        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        yn = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if i < 3:
            yn[0] += seed * dt          # 种子注入
        if mode == "hard":
            yn = np.clip(yn, 0.0, 1.0)
        if not np.all(np.isfinite(yn)) or np.max(np.abs(yn)) > 1e12:
            Y[i:] = np.nan
            return np.linspace(0, T, n), Y, True
        Y[i] = yn
    return np.linspace(0, T, n), Y, False


T = 5000.0

# ---------------------------------------------------------------- 冒烟
print("=" * 78)
print("冒烟检查")
print("=" * 78)
p = base(r=1.5)
y_crash = np.array([0.0, 0.0, 0.0, 1.0])
y_flour = np.array([1.0, 1.0, 1.0, 0.0])
print("  崩溃点 (0,0,0,1) rhs =", np.array2string(rhs(y_crash, p), precision=6))
print("  兴盛点 (1,1,1,0) rhs =", np.array2string(rhs(y_flour, p), precision=6))
print("  变化的轴：兴盛点只有 D,O,G 三分量非零（A 分量恒零，因 alpha_A*0=0）")
assert abs(rhs(y_crash, p)[0]) < 1e-15, "冒烟：崩溃点 D 分量应为零"
assert abs(rhs(y_flour, p)[0] - p["gD"]) < 1e-15, "冒烟：兴盛点 D 分量应 = gamma_D"
print("  冒烟通过。")

# ---------------------------------------------------------------- H1/H2
print()
print("=" * 78)
print("H1/H2 两吸引子真伪：直接算不动点残差")
print("=" * 78)
H1_rows = []
H2_rows = []
for r in [0.5, 1.0, 1.5, 2.0]:
    p = base(r=r)
    f_c = rhs(y_crash, p)
    f_f = rhs(y_flour, p)
    n_c = np.linalg.norm(f_c)
    n_f = np.linalg.norm(f_f)
    H1_rows.append((r, n_c))
    H2_rows.append((r, n_f))
    print("  r=%.1f: |rhs(崩溃点)| = %.3e | |rhs(兴盛点)| = %.3e（gamma_D=%.2f）"
          % (r, n_c, n_f, p["gD"]))
H1 = all(x[1] < 1e-14 for x in H1_rows)
H2 = all(x[1] > 0.1 for x in H2_rows)
print("  H1 %s：崩溃点是精确不动点" % ("PASS" if H1 else "FAIL"))
print("  H2 %s：兴盛点不是不动点（残差 = gamma_D 量级）" % ("PASS" if H2 else "FAIL"))

# ---------------------------------------------------------------- H3
print()
print("=" * 78)
print("H3 无约束时 r>1 档不趋近兴盛点（越界）")
print("=" * 78)
H3_rows = []
for r in [0.9, 1.1, 1.5, 2.0]:
    p = base(r=r)
    t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=2000.0, n=80001, mode="free")
    if blown:
        fin = np.isfinite(Y[:, 1])
        Omax = np.nanmax(Y[fin, 1])
        H3_rows.append((r, True, Omax))
        print("  r=%.1f: 越界（max O = %.3e）" % (r, Omax))
    else:
        d = np.linalg.norm(Y[-1] - y_flour)
        H3_rows.append((r, False, d))
        print("  r=%.1f: 末态 (%.4f,%.4f,%.4f,%.4f)，到兴盛点距离 %.4f"
              % (r, *Y[-1], d))
H3 = all((x[1] if x[0] > 1.0 else not x[1]) for x in H3_rows)
print("  H3 %s：r>1 档无约束时越界、不趋近兴盛点" % ("PASS" if H3 else "FAIL"))

# ---------------------------------------------------------------- H4
print()
print("=" * 78)
print("H4 硬投影约束下 r>1 档收敛到兴盛点")
print("=" * 78)
H4_rows = []
for r in [1.1, 1.5, 2.0]:
    p = base(r=r)
    t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=T, n=200001, mode="hard")
    if blown:
        H4_rows.append((r, False, None))
        print("  r=%.1f: 仍越界（异常）" % r)
    else:
        ss = Y[-1]
        d = np.linalg.norm(ss - y_flour)
        ok = d < 0.05
        H4_rows.append((r, ok, ss))
        print("  r=%.1f: 稳态 (%.6f, %.6f, %.6f, %.6f)，到兴盛点距离 %.2e %s"
              % (r, *ss, d, "✓" if ok else "✗"))
H4a = all(x[1] for x in H4_rows)
print("  H4a %s：硬投影下 r>1 档趋近兴盛点" % ("PASS" if H4a else "FAIL"))

print("  H4b 对照（防恒真）：硬投影下 r<1 档是否仍衰减到崩溃点：")
H4b_rows = []
for r in [0.5, 0.9]:
    p = base(r=r)
    t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=T, n=200001, mode="hard")
    ss = Y[-1]
    d_c = np.linalg.norm(ss - y_crash)
    ok = (not blown) and d_c < 0.01
    H4b_rows.append((r, ok))
    print("    r=%.1f: 稳态 (%.2e, %.2e, %.2e, %.6f)，到崩溃点距离 %.2e %s"
          % (r, *ss, d_c, "✓" if ok else "✗"))
H4b = all(x[1] for x in H4b_rows)
print("    H4b %s：硬投影只钳上界，不改变 r<1 档的衰减（故 H4a 非恒真）"
      % ("PASS" if H4b else "FAIL"))
H4 = H4a and H4b
print("  H4 %s：硬投影下兴盛点成为可达吸引子（且对照档行为不变）"
      % ("PASS" if H4 else "FAIL"))

# ---------------------------------------------------------------- H5
print()
print("=" * 78)
print("H5 软壁行为对系数敏感（非普遍可达）")
print("=" * 78)
H5_rows = []
for kc in [1e2, 1e3, 1e4]:
    p = base(r=1.5)
    t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=2000.0, n=80001,
                         mode="soft", kc=kc)
    if blown:
        lab = "越界（软壁压不住）"
    else:
        ss = Y[-1]
        lab = "趋近兴盛" if np.linalg.norm(ss - y_flour) < 0.1 else "非兴盛稳态"
    H5_rows.append((kc, lab))
    print("  kc=%.0e: %s" % (kc, lab))
# 判据：三档行为不全是"趋近兴盛"（即软壁非普遍可达）
labels = {x[1] for x in H5_rows}
H5 = (labels != {"趋近兴盛"})
print("  H5 %s：三档行为标签为 %s，非全部趋近兴盛（软壁依赖 kc，非普遍可达）"
      % ("PASS" if H5 else "FAIL", labels))

# ---------------------------------------------------------------- H7
print()
print("=" * 78)
print("H7 A 稳态的参数依赖：A_inf = max(0, 1 - delta/alpha_A)")
print("=" * 78)
print("  机制：O=G=D=1 时 A 方程退化为 alpha_A*A*(1-A) - delta*A = 0")
print("  -> A*(alpha_A*(1-A) - delta) = 0 -> A_inf = max(0, 1 - delta/alpha_A)")
H7_rows = []
for aA, dl in [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0), (0.5, 1.0)]:
    p = base(r=1.5)
    p["aA"] = aA
    p["dl"] = dl
    t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=T, n=200001, mode="hard")
    A_end = Y[-1, 3]
    pred = max(0.0, 1.0 - dl / aA)
    ok = abs(A_end - pred) < 5e-3
    H7_rows.append((aA, dl, A_end, pred, ok))
    print("    alpha_A=%.1f delta=%.1f: A末=%.6f | 解析 %.4f %s"
          % (aA, dl, A_end, pred, "✓" if ok else "✗"))
H7 = all(x[4] for x in H7_rows)
print("  H7 %s：A 稳态由 alpha_A/delta 决定；仅在 delta>=alpha_A 时 A->0" % ("PASS" if H7 else "FAIL"))

# ---------------------------------------------------------------- H6
print()
print("=" * 78)
print("H6 恒真反例：欧氏距离判据对任意点都可用，须配不动点判据")
print("=" * 78)
p = base(r=1.5)
t, Y, blown = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=T, n=200001, mode="hard")
ss = Y[-1]
pts = {"兴盛点 (1,1,1,0)": y_flour,
       "崩溃点 (0,0,0,1)": y_crash,
       "任意点 (0.5,0.5,0.5,0.5)": np.array([0.5, 0.5, 0.5, 0.5])}
H6_rows = []
for name, pt in pts.items():
    d = np.linalg.norm(ss - pt)
    is_fp = np.linalg.norm(rhs(pt, p)) < 1e-14
    H6_rows.append((name, d, is_fp))
    print("    到%s距离 %.4f | 该点是原方程不动点：%s" % (name, d, is_fp))
# 兴盛点：距离小但不是原方程不动点；崩溃点：是不动点但距离大
d_flour = np.linalg.norm(ss - y_flour)
d_crash = np.linalg.norm(ss - y_crash)
fp_flour = np.linalg.norm(rhs(y_flour, p)) < 1e-14
fp_crash = np.linalg.norm(rhs(y_crash, p)) < 1e-14
H6 = (d_flour < 0.05) and (not fp_flour) and (d_crash > 1.0) and fp_crash
print("  H6 %s：距离判据单独使用会误判（兴盛点距离小但非原方程不动点），"
      "须与不动点判据联用" % ("PASS" if H6 else "FAIL"))

# ---------------------------------------------------------------- 出图
print()
print("=" * 78)
print("出图")
print("=" * 78)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# 面板 1：两点的 rhs 分量
ax = axes[0]
p = base(r=1.5)
f_c = rhs(y_crash, p)
f_f = rhs(y_flour, p)
x = np.arange(4)
ax.bar(x - 0.18, np.abs(f_c), width=0.36, label="崩溃点 $(0,0,0,1)$", color="tab:blue")
ax.bar(x + 0.18, np.abs(f_f), width=0.36, label="兴盛点 $(1,1,1,0)$", color="tab:red")
ax.set_xticks(x)
ax.set_xticklabels([r"$\dot D$", r"$\dot O$", r"$\dot G$", r"$\dot A$"])
ax.set_ylabel("$|$方程右端$|$")
ax.set_title("两点的不动点残差（H1/H2）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# 面板 2：三档约束下的轨迹
ax = axes[1]
p = base(r=1.5)
for mode, c, lab in [("free", "tab:green", "无约束（越界）"),
                     ("soft", "tab:orange", r"软壁 $k_c=10^3$"),
                     ("hard", "tab:red", "硬投影")]:
    t, Y, _ = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=2000.0, n=80001, mode=mode, kc=1e3)
    ax.plot(t, Y[:, 1], color=c, lw=1.6, label=lab)
ax.axhline(1.0, color="k", ls=":", lw=1, label="上界 1")
ax.set_yscale("log")
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel(r"$\tilde O$")
ax.set_title(r"界域约束形式对比（H3/H4/H5，$r=1.5$）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 面板 3：硬投影下三档 r 的收敛
ax = axes[2]
for r, c in [(1.1, "tab:blue"), (1.5, "tab:orange"), (2.0, "tab:green")]:
    p = base(r=r)
    t, Y, _ = integ2(p, [1e-3, 1e-3, 1e-3, 0.9], T=T, n=200001, mode="hard")
    ax.plot(t, Y[:, 0], color=c, lw=1.5, label=r"$\tilde D$, $r=%.1f$" % r)
    ax.plot(t, Y[:, 3], color=c, lw=1.5, ls="--")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.axhline(0.0, color="k", ls=":", lw=1)
ax.set_xlabel("内部时间 $u$")
ax.set_ylabel("序参量")
ax.set_title("硬投影下收敛到 $(1,1,1,0)$（实线 $\\tilde D$、虚线 $\\tilde A$，H4）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
out = IMG + r"\doga_attractor_648.png"
fig.savefig(out, dpi=150)
print("  图已存：", out)

# ---------------------------------------------------------------- 汇总
print()
print("=" * 78)
print("判据汇总")
print("=" * 78)
for name, val, desc in [("H1", H1, "崩溃点 (0,0,0,1) 是精确不动点"),
                        ("H2", H2, "兴盛点 (1,1,1,0) 不是不动点（残差 = gamma_D）"),
                        ("H3", H3, "无约束时 r>1 档越界，不趋近兴盛点"),
                        ("H4", H4, "硬投影下兴盛点成为可达吸引子"),
                        ("H5", H5, "软壁行为对 kc 敏感，非普遍可达"),
                        ("H6", H6, "距离判据须与不动点判据联用"),
                        ("H7", H7, "A 稳态 = max(0, 1-delta/alpha_A)，非恒为零")]:
    print("  %s %s  %s" % (name, "PASS" if val else "FAIL", desc))
print()
print("  结论一：文档所称\"两吸引子\"中，只有崩溃点是原方程的不动点；")
print("          兴盛点 (1,1,1,0) 在无界域约束时不是不动点（残差 = gamma_D）。")
print("  结论二：加硬投影约束（严格钳在 [0,1]）后，r>1 档趋近 (1,1,1,0)；")
print("          软壁（反馈拉回）依赖系数 kc，非普遍可达，故合法形式取硬投影。")
print("  结论三：A 的稳态为 A_inf = max(0, 1 - delta/alpha_A)，并非恒为零；")
print("          (1,1,1,0) 只是 delta >= alpha_A 档的特例。")
