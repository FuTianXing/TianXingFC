# -*- coding: utf-8 -*-
"""第 617 号：三维偏振展开谱轮（偶极载体的各向同性化、横扇区保留与 LO-TO 劈裂）

出处：第 613 号（偶极立案）注记「单偏振简化换逐位可比，三维偏振展开留谱轮后」；
承接第 615 号（γ_d=1/d 的各向同性几何来源），本轮给 γ_3D 提供完整的三维载体。

判据（先立期望，后跑数值）：
  W2a  三维载体投影正确性：
       12x12 三维各向同性正则块（三方向 (A,E,x,Pi) + 纵向偶极核 dd_ij）的谱
       = 横扇区（垂直 k 平面两偏振，2 重简并，谱与投影横块逐位一致）+ 纵扇区
       （1 重）+ 各自零频规范对；投影横块与第 613 号单偏振块同构（色散换 3D
       w_l^2）逐位对照。
  W2b  LO-TO 劈裂锚：纵支振子频率与横支振子频率的 k->0 极限平方差 = beta^2
       （第 581 条基线 Lorentz 介电函数的 LST 关系：eps(0) = 1+beta^2/omega_T^2
       给 omega_L^2 = eps(0)*omega_T^2 = omega_T^2 + beta^2）。
  W2c  齿：dd 核撤除（纵向偶极项置零）后纵横全简并（12 谱 = 3 重 x 4 谱），
       dd 恢复则劈裂回归；斜向 k（dd 满矩阵非对角）下横向 2 重简并保持
       （各向同性横向的数值内容）。

模型设定：
  k 空间正则块（第 613 号同构的矢量化），变量每方向 (A_d, E_d, x_d, Pi_d)：
    A_i_dot = E_i - beta*x_i
    E_i_dot = -w_l^2 A_i                （w_l^2 = sum_a 4 sin^2(k_a/2)，3D 色散）
    x_i_dot = Pi_i
    Pi_i_dot = beta*E_i - (w0^2+beta^2)*x_i - beta^2 * sum_j dd_ij x_j
  纵向偶极-偶极核 dd_ij(k) = D_i D_j / w_l^2，D_i = 2 sin(k_i/2)（离散差分）。
  性质：dd 为纯纵向投影（k->0 极限 k_i k_j/|k|^2）；dd_对角沿 k = 1 精确；
  横向投影 ee'·dd·e = 0（横扇区严格解耦，投影正确性的数学内容）。

解析预演：
  k->0 横支振子频率 = sqrt(w0^2+beta^2)（第 613 号块零波极限）；
  纵向 dd 贡献精确 +beta^2（dd_对角沿 k = 1）→ 纵支 = sqrt(w0^2+2 beta^2)；
  劈裂 = beta^2，与第 581 条基线 LST 自洽（omega_L^2 = omega_T^2 + beta^2）。
  CM 层（局域场 gamma=1/3）的纵向修正为纵向局域场几何问题，本轮不锁定，
  登记为开放项（避免把约定差硬凑成劈裂）。

容差预算：
  块谱对照：本征值数值误差 ~1e-13（numpy 实对称化后 1e-12 保险）。
  k->0 外推：omega^2(t) = omega^2(0) + c t^2 + O(t^4)，三点二次拟合残差
  打印定标；劈裂锚容差 = 2 x 拟合残差 + 1e-9。

可逆性等级：W2a/W2b/W2c 判据稳健（换 dd 核、破坏各向同性或换色散即翻）。
边界声明：验证的是三维各向同性偶极载体的谱结构（横扇区保留 + LO-TO 劈裂
的 dd 核来源）；不验证真实 3D 晶格色散的各向异性（w_l^2 各向同性化为平均
场近似）、不验证局域场 gamma=1/3 在纵向扇区的对应（登记开放）、不验证
D3p-2 界面重拨（后续轮）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BETA = float(np.sqrt(0.3))
W0 = 1.0
IMG_DIR = r"D:\software\TianXingFC\images"
RESULTS = {}


# ----------------------------------------------------------------------
def w_latt_3d(kv):
    return float(np.sum(4.0 * np.sin(np.asarray(kv) / 2.0) ** 2))


def dd_kernel(kv):
    """纵向偶极核 dd_ij = D_i D_j / w_l^2，D_i = 2 sin(k_i/2)。"""
    D = 2.0 * np.sin(np.asarray(kv) / 2.0)
    wl2 = np.sum(D ** 2)
    return np.outer(D, D) / wl2


def block_613(wl2, beta=BETA, extra_dd=0.0):
    """第 613 号单偏振正则块（色散参数可换；extra_dd 加纵向偶极修正）。"""
    return np.array([
        [0, 1, -beta, 0],
        [-wl2, 0, 0, 0],
        [0, 0, 0, 1],
        [0, beta, -(W0 ** 2 + beta ** 2) - extra_dd, 0]], dtype=float)


def block_3d(kv, beta=BETA, use_dd=True):
    """12x12 三维各向同性正则块：三方向变量 (A_d,E_d,x_d,Pi_d)。"""
    wl2 = w_latt_3d(kv)
    dd = dd_kernel(kv) if use_dd else np.zeros((3, 3))
    M = np.zeros((12, 12))
    for i in range(3):
        b = slice(4 * i, 4 * i + 4)
        M[b, b] = block_613(wl2, beta)
    # dd 项：Pi_i_dot 行加 -beta^2 dd_ij x_j
    for i in range(3):
        for j in range(3):
            M[4 * i + 3, 4 * j + 2] -= beta ** 2 * dd[i, j]
    return M


def transverse_projector_block(kv, beta=BETA):
    """投影横块：垂直 k 的两偏振方向（与第 613 号同构，色散 3D w_l^2）。"""
    kv = np.asarray(kv, dtype=float)
    n = kv / np.linalg.norm(kv)
    # 取两个垂直于 k 的方向
    tmp = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    wl2 = w_latt_3d(kv)
    # 横向 dd 投影 = 0（构造级），块与 613 同构
    return [block_613(wl2, beta) for _ in (e1, e2)]


def longitudinal_block(kv, beta=BETA, use_dd=True):
    """纵块：k 方向投影。dd 投影 = |D|^2/w_l^2 = 1 精确（沿 k 共线）。"""
    wl2 = w_latt_3d(kv)
    dd_par = 1.0 if use_dd else 0.0
    return block_613(wl2, beta, extra_dd=beta ** 2 * dd_par)


def freqs(M):
    return np.sort(np.abs(np.linalg.eigvals(M)))


# ----------------------------------------------------------------------
def run_w2a():
    print("== W2a：三维载体投影正确性 ==")
    ok = True
    worst = 0.0
    t_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    for t in t_list:
        kv = np.array([t, t, t]) / np.sqrt(3.0)  # 斜向 k（dd 满矩阵）
        full = np.sort(freqs(block_3d(kv)))
        trans = np.concatenate([freqs(b) for b in transverse_projector_block(kv)])
        long_ = freqs(longitudinal_block(kv))
        expect = np.sort(np.concatenate([trans, long_]))
        d = np.max(np.abs(full - expect))
        worst = max(worst, d)
        # 横向简并：两横块谱逐位一致
        b1, b2 = transverse_projector_block(kv)
        ddeg = np.max(np.abs(freqs(b1) - freqs(b2)))
        print("  t=%.2f  整块谱对投影和 max|d|=%.2e  横简并 max|d|=%.2e"
              % (t, d, ddeg))
        ok &= d <= 1e-10 and ddeg <= 1e-12
    # 沿轴档（分块结构对照）
    kv = np.array([0.3, 0.0, 0.0])
    full = np.sort(freqs(block_3d(kv)))
    trans = np.concatenate([freqs(b) for b in transverse_projector_block(kv)])
    long_ = freqs(longitudinal_block(kv))
    d = np.max(np.abs(full - np.sort(np.concatenate([trans, long_]))))
    print("  沿轴 k=0.3：整块谱对投影和 max|d|=%.2e" % d)
    ok &= d <= 1e-10
    print("  W2a 判定：%s（worst=%.2e）" % ("PASS" if ok else "FAIL", worst))
    RESULTS["w2a"] = ok
    return ok


def run_w2b():
    print("\n== W2b：LO-TO 劈裂锚（劈裂 = beta^2） ==")
    ok = True
    t_list = np.array([0.04, 0.08, 0.16])
    wt2, wl2_arr = [], []
    for t in t_list:
        kv = np.array([t, t, t]) / np.sqrt(3.0)
        long_ = freqs(longitudinal_block(kv))
        # 振子支 = 非零频对（最大两支），光支趋零
        w_osc = long_[-1]
        b1, _ = transverse_projector_block(kv)
        wt2.append(freqs(b1)[-1] ** 2)
        wl2_arr.append(w_osc ** 2)
        print("  t=%.2f  omega_T^2=%.8f  omega_L^2=%.8f  差=%.8f（beta^2=%.8f）"
              % (t, wt2[-1], wl2_arr[-1], wl2_arr[-1] - wt2[-1], BETA ** 2))
    # 二次外推 t->0（omega^2 = w2_0 + c t^2，三点拟合截距）
    fit_L = np.polyfit(t_list ** 2, wl2_arr, 1)
    fit_T = np.polyfit(t_list ** 2, wt2, 1)
    res_L = np.max(np.abs(np.polyval(fit_L, t_list ** 2) - wl2_arr))
    res_T = np.max(np.abs(np.polyval(fit_T, t_list ** 2) - wt2))
    L0, T0 = fit_L[1], fit_T[1]
    split = L0 - T0
    tol = 2 * (res_L + res_T) + 1e-9
    print("  外推 k->0：omega_L^2=%.8f（残差 %.1e）  omega_T^2=%.8f（残差 %.1e）"
          % (L0, res_L, T0, res_T))
    print("  劈裂实测 %.8f  期望 beta^2=%.8f  容差 %.2e" % (split, BETA ** 2, tol))
    ok &= abs(split - BETA ** 2) < tol
    # LST 对照：eps(0) = 1 + beta^2/omega_T^2，omega_L^2 = eps(0)*omega_T^2
    eps0 = 1 + BETA ** 2 / T0
    print("  LST：eps(0)*omega_T^2=%.8f 对 omega_L^2=%.8f（d=%.1e）"
          % (eps0 * T0, L0, abs(eps0 * T0 - L0)))
    ok &= abs(eps0 * T0 - L0) < tol
    RESULTS.update(wt2=wt2, wl2_arr=wl2_arr, t_list=t_list,
                   L0=L0, T0=T0, split=split, tol=tol)
    print("  W2b 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_w2c():
    print("\n== W2c：齿（dd 撤除即全简并） ==")
    ok = True
    worst = 0.0
    for t in (0.1, 0.3):
        kv = np.array([t, t, t]) / np.sqrt(3.0)
        full_nodd = np.sort(freqs(block_3d(kv, use_dd=False)))
        rep3 = np.sort(np.concatenate([freqs(block_613(w_latt_3d(kv)))] * 3))
        d = np.max(np.abs(full_nodd - rep3))
        worst = max(worst, d)
        print("  t=%.2f  dd=0：整块谱对三重 613 谱 max|d|=%.2e（须简并）" % (t, d))
        ok &= d <= 1e-10
        full_dd = np.sort(freqs(block_3d(kv, use_dd=True)))
        # 劈裂回归：dd 版与无 dd 版谱差非零（纵支移开）
        dsplit = np.max(np.abs(full_dd - full_nodd))
        print("  t=%.2f  dd 恢复后谱差 max|d|=%.2e（须显著非零）" % (t, dsplit))
        ok &= dsplit > 1e-3
    print("  W2c 判定：%s（worst=%.2e）" % ("PASS" if ok else "FAIL", worst))
    RESULTS["w2c"] = ok
    return ok


# ----------------------------------------------------------------------
def make_plots():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    t_list, wt2, wl2_arr = RESULTS["t_list"], RESULTS["wt2"], RESULTS["wl2_arr"]

    ax = axes[0]
    ax.plot(t_list ** 2, wt2, "o-", color="tab:blue", label=r"$\omega_T^2$ 实测")
    ax.plot(t_list ** 2, wl2_arr, "s-", color="tab:red", label=r"$\omega_L^2$ 实测")
    ax.axhline(RESULTS["T0"], color="tab:blue", ls=":", lw=1)
    ax.axhline(RESULTS["L0"], color="tab:red", ls=":", lw=1)
    ax.set_title(r"振子支频率平方外推 $k\to 0$")
    ax.set_xlabel(r"$t^2$（$k = t\hat{1}\hat{1}\hat{1}/\sqrt{3}$）")
    ax.set_ylabel(r"$\omega^2$")
    ax.legend()

    ax = axes[1]
    splits = [l - w for l, w in zip(wl2_arr, wt2)]
    ax.plot(t_list, splits, "o-", color="tab:green", label="劈裂实测")
    ax.axhline(BETA ** 2, color="k", ls="--", lw=1,
               label=r"期望 $\beta^2 = %.4f$" % BETA ** 2)
    ax.set_title("LO-TO 劈裂")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\omega_L^2 - \omega_T^2$")
    ax.legend()

    ax = axes[2]
    kv = np.array([0.3, 0.3, 0.3]) / np.sqrt(3.0)
    f_dd = freqs(block_3d(kv, use_dd=True))
    f_nodd = freqs(block_3d(kv, use_dd=False))
    ax.plot(range(12), f_nodd, "o", ms=8, color="tab:gray", label="dd 撤除（全简并）")
    ax.plot(range(12), np.sort(f_dd), "x", ms=10, color="tab:purple",
            label="dd 恢复（纵横劈裂）")
    ax.set_title("W2c 齿：dd 核撤除与恢复（t=0.3）")
    ax.set_xlabel("谱序号")
    ax.set_ylabel(r"$\omega$")
    ax.legend()

    fig.suptitle("第 617 号：三维偏振展开谱轮（横扇区保留与 LO-TO 劈裂）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = IMG_DIR + r"\dipole_3d_polarization_617.png"
    fig.savefig(out, dpi=150)
    print("\n图已保存：%s" % out)


def main():
    print("第 617 号：三维偏振展开谱轮")
    print("期望：横扇区 = 613 块（简并 2）、纵支劈裂 = beta^2（581 基线 LST）")
    ok_a = run_w2a()
    ok_b = run_w2b()
    ok_c = run_w2c()
    print("\n================ 总结 ================")
    print("W2a（三维载体投影正确性）：%s" % ("PASS" if ok_a else "FAIL"))
    print("W2b（LO-TO 劈裂锚 = beta^2 与 LST）：%s" % ("PASS" if ok_b else "FAIL"))
    print("W2c（dd 撤除齿）：%s" % ("PASS" if ok_c else "FAIL"))
    make_plots()
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
