# -*- coding: utf-8 -*-
"""第 440 条：轴向偏折的判决性测量（两条色散路线的方向律与差异恒等式）。

判据：
  J1 连续时间路线（原型层）的方向依赖系数 A_cont = -F/24，方向比 6:3:2；
  J2 继承规则（采样层）的方向依赖系数 A_inh = +(1-F)/24，方向比 0:3:4（轴向精确为零）；
  J3 两路线之差 (A_inh - A_cont) 在任意方向恒为 1/24（方向无关的差异恒等式）。
  附：轴向色散的两路线数值演化核验（过零测频）。

约定：lP = tP = 1（自然单位）；波长与 k 无量纲；方向取轴向 (1,0,0)、面对角 (1,1,0)/sqrt2、
体对角 (1,1,1)/sqrt3，对应 F = sum_a k_a^4 / k^4 = 1、1/2、1/3。

容差预算：A 的提取用两项拟合（(lP k)^2 与 (lP k)^4），取 lP k <= 2e-3 时截断项量级 4e-6 但被
第二项吸收，残差量级 1e-8；故容差取 1e-6。J2 的轴向零为代数恒等式，容差取 1e-14。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_axial_dispersion.png"
lP, tP = 1.0, 1.0
c = lP / tP

dirs = {
    "axis": np.array([1.0, 0.0, 0.0]),
    "face": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    "body": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
}

def F_of(kvec):
    k = np.linalg.norm(kvec)
    return float(np.sum(kvec ** 4) / k ** 4)

def S_of(kvec):
    return float(np.sum(np.sin(kvec * lP / 2.0) ** 2))

def omega_cont(kvec):
    return 2.0 * np.sqrt(S_of(kvec)) / tP

def omega_inh(kvec):
    s = np.sqrt(S_of(kvec))
    s = min(s, 1.0)
    return 2.0 * np.arcsin(s) / tP

def extract_A(fun, kvec_unit):
    ks = np.array([0.5e-3, 1.0e-3, 2.0e-3])
    Rs = []
    for kk in ks:
        kvec = kvec_unit * kk
        w = fun(kvec)
        k = np.linalg.norm(kvec)
        Rs.append(w / (c * k) - 1.0)
    Rs = np.array(Rs)
    M = np.vstack([(ks * lP) ** 2, (ks * lP) ** 4]).T
    sol, *_ = np.linalg.lstsq(M, Rs, rcond=None)
    return float(sol[0])

rows = []
for name, u in dirs.items():
    F = F_of(u * 1e-3)
    Ac = extract_A(omega_cont, u)
    Ai = extract_A(omega_inh, u)
    rows.append((name, F, Ac, -F / 24.0, Ai, (1.0 - F) / 24.0, Ai - Ac))

# 数值演化核验（轴向，过零测频）
def omega_verlet(lam, dt=2e-3, n=400000):
    w = np.sqrt(lam) / tP
    x, v = 1.0, 0.0
    prev, cross = x, []
    for i in range(n):
        a = -(w ** 2) * x
        v += a * dt
        x += v * dt
        if prev > 0.0 >= x:
            cross.append(i * dt)
        prev = x
        if len(cross) >= 5:
            break
    return 2.0 * np.pi / np.mean(np.diff(cross))

def omega_recur(lam, n=20000):
    p2, p1 = 1.0, 1.0
    cross = []
    for s in range(2, n):
        a2 = 2.0 * p1 - p2 - lam * p1
        if p1 > 0.0 >= a2:
            cross.append(s)
        p2, p1 = p1, a2
        if len(cross) >= 5:
            break
    T = np.mean(np.diff(cross))           # 下降过零每周期一次
    return 2.0 * np.pi / T

k0 = 0.1
kv = dirs["axis"] * k0
lam0 = 4.0 * S_of(kv)
wc_num, wi_num = omega_verlet(lam0), omega_recur(lam0)
wc_ana, wi_ana = omega_cont(kv), omega_inh(kv)

print("J1/J2 方向依赖系数（A 为 (lP k)^2 阶相对修正系数）")
print("方向      F       连续A实测    连续期望    继承A实测    继承期望    差(A继承-A连续)")
for r in rows:
    print("%-7s %.4f  %+.8f  %+.8f  %+.8f  %+.8f  %+.8f" % r)
print()
print("轴向数值演化核验（lP k = 0.1）")
print("连续：Verlet 实测 omega = %.8f，解析 = %.8f，相对差 %.2e" % (wc_num, wc_ana, abs(wc_num - wc_ana) / wc_ana))
print("继承：递推实测 omega = %.8f，解析 = %.8f，相对差 %.2e" % (wi_num, wi_ana, abs(wi_num - wi_ana) / wi_ana))
print()
flag1 = bool(all(abs(r[2] - r[3]) < 1e-6 for r in rows))
flag2 = bool(all(abs(r[4] - r[5]) < 1e-6 for r in rows) and abs(rows[0][4]) < 1e-14)
flag3 = bool(all(abs(r[6] - 1.0 / 24.0) < 1e-6 for r in rows))
print("判据 J1（连续时间路线方向律 -F/24）：%s" % ("通过" if flag1 else "未通过"))
print("判据 J2（继承规则方向律 +(1-F)/24，轴向零）：%s" % ("通过" if flag2 else "未通过"))
print("判据 J3（差异恒等式 1/24，方向无关）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

# 出图
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
kk = np.linspace(1e-4, np.pi - 1e-3, 500)
wc = [2.0 * np.sin(k / 2.0) for k in kk]
wi = [k for k in kk]
ax[0, 0].plot(kk, wc, "b-", lw=1.2, label="continuous (prototype): 2 sin(k/2)")
ax[0, 0].plot(kk, wi, "g-", lw=1.2, label="inheritance (sampling): k (exact linear)")
ax[0, 0].plot(kk, kk, "k--", lw=0.8, label="ck")
ax[0, 0].set_xlabel("lP k (axis)")
ax[0, 0].set_ylabel("omega")
ax[0, 0].set_title("axial dispersion to Brillouin edge")
ax[0, 0].legend(fontsize=7)

xx = np.arange(3)
ax[0, 1].bar(xx - 0.2, [r[2] for r in rows], 0.4, color="steelblue", label="continuous")
ax[0, 1].bar(xx + 0.2, [r[4] for r in rows], 0.4, color="seagreen", label="inheritance")
ax[0, 1].axhline(0.0, color="k", lw=0.8)
ax[0, 1].set_xticks(xx); ax[0, 1].set_xticklabels(["axis", "face", "body"], fontsize=8)
ax[0, 1].set_ylabel("A (relative (lP k)^2 coefficient)")
ax[0, 1].set_title("J1/J2 direction-dependent coefficient")
ax[0, 1].legend(fontsize=7)

ax[1, 0].bar(xx, [r[6] for r in rows], color="indianred")
ax[1, 0].axhline(1.0 / 24.0, color="k", ls="--", lw=1.0, label="1/24")
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(["axis", "face", "body"], fontsize=8)
ax[1, 0].set_ylabel("A_inh - A_cont")
ax[1, 0].set_title("J3 difference identity")
ax[1, 0].legend(fontsize=7)

ratio_c = np.array([abs(r[2]) for r in rows]); ratio_c = ratio_c / ratio_c[0]
ratio_i = np.array([abs(r[4]) for r in rows]); ratio_i = ratio_i / ratio_i.max()
ax[1, 1].plot(xx, ratio_c, "bo-", label="continuous (shape 6:3:2)")
ax[1, 1].plot(xx, ratio_i, "gs-", label="inheritance (shape 0:3:4)")
ax[1, 1].set_xticks(xx); ax[1, 1].set_xticklabels(["axis", "face", "body"], fontsize=8)
ax[1, 1].set_ylabel("normalized magnitude")
ax[1, 1].set_title("direction law comparison")
ax[1, 1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
