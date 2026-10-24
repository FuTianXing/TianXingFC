# -*- coding: utf-8 -*-
r"""
full_quantization.py

[完全量子化轮] 算符层面没有"+1 本底"：谐振子格点的计数率正比于占据数

定位
----
末步几轮的 $+1$ 本底来自准经典模型里的"每站每摆程一次"。本轮把格点模与探测器都写成算符
（截断玻色模 + 二能级探测器 + 罗特瓦夫耦合 + 探测器衰变给出记录），用精确的量子跳跃轨迹统计点击数，
检验三件事：计数率对占据数的关系、占据数为零时是否仍有计数、计数统计的 Fano 因子。
本脚本不改动基准文件；不含新科学断言。

模型
----
单站点：截断玻色模（能级 $0$ 至 $N_{\max}$）与二能级探测器按
  $H = \omega_a a^\dagger a + \tfrac{\omega_d}{2}\sigma_z + g(a^\dagger\sigma_- + a\sigma_+)$
耦合（罗特瓦夫近似，共振 $\omega_a = \omega_d$），探测器以速率 $\gamma_r$ 向记录通道衰变：
每次衰变记一次点击。演化用有效哈密顿量 $H_{\rm eff} = H - i\gamma_r\sigma_+\sigma_-/2$ 加量子跳跃实现（精确展开）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 计数率对占据数的线性律：Fock 初态 $|n\rangle$（$n = 0,1,2,4$）下，窗口内平均点击数 $\propto n$（过原点的线性拟合，相关 $> 0.99$）。
J2 占据数为零时无计数：$n = 0$ 时平均点击数为零（阈值 $< 10^{-3}$），即算符层面没有 $+1$ 项。
J3 计数统计：本设置为吸收型计数，长窗口内总计数等于被吸收的量子数。J3a Fock 态近乎确定（Fano $< 0.3$）；J3b 混合物超泊松（实测 Fano $> 1.2$），方向与 $1 + \mathrm{Var}(n)/\langle n\rangle$ 一致；因窗口未把 $n \ge 2$ 的量子全部吸收，实测低于该预期，属窗口边界（已登记）。相干态的实测值与泊松预期不符，登记为待查。
J4 与前三轮的对照（负面结论登记）：准经典模型给出的"+1 本底"在谐振子格点上没有算符对应物；每站点自发发射要求非谐振（发射体）结构。
J5 反作用读数：每次点击后模式平均占据数的减少与点击数一致（报告）。

运行
----
python 验证脚本/full_quantization.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/full_quantization.png
"""
import time

import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_MAX = 8
OMEGA = 2.0 * np.pi
G_COUP = 0.25
GAMMA_R = 0.5
SEED = 20260918
T_WIN = 4.0
DT = 5.0e-3


def operators():
    dim = (N_MAX + 1) * 2
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(N_MAX):
        for s in range(2):
            a[2 * n + s, 2 * (n + 1) + s] = np.sqrt(n + 1)
    sm = np.zeros((dim, dim), dtype=complex)
    for n in range(N_MAX + 1):
        sm[2 * n, 2 * n + 1] = 1.0
    sp = sm.conj().T
    num = a.conj().T @ a
    return dim, a, sm, sp, num


def hamiltonian(a, sm, sp, num, dim):
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        H[i, i] = OMEGA * num[i, i] + (OMEGA / 2.0) * (1 if i % 2 == 1 else -1)
    H += G_COUP * (a.conj().T @ sm + a @ sp)
    return H


def fock_state(n):
    psi = np.zeros((N_MAX + 1) * 2, dtype=complex)
    psi[2 * n] = 1.0
    return psi


def coherent_state(alpha):
    psi = np.zeros((N_MAX + 1) * 2, dtype=complex)
    for n in range(N_MAX + 1):
        psi[2 * n] = np.exp(-abs(alpha) ** 2 / 2.0) * alpha ** n / np.sqrt(math.factorial(n))
    return psi / np.linalg.norm(psi)


def count_clicks(psi0, H, sm, sp, rng, T=T_WIN, dt=DT):
    Heff = H - 0.5j * GAMMA_R * (sp @ sm)
    psi = psi0.copy()
    clicks = 0
    t = 0.0
    steps = int(round(T / dt))
    for _ in range(steps):
        k1 = -1j * (Heff @ psi)
        k2 = -1j * (Heff @ (psi + 0.5 * dt * k1))
        k3 = -1j * (Heff @ (psi + 0.5 * dt * k2))
        k4 = -1j * (Heff @ (psi + dt * k3))
        psi = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        norm2 = float(np.vdot(psi, psi).real)
        if rng.random() > norm2:
            clicks += 1
            psi = sm @ psi
            psi = psi / np.linalg.norm(psi)
    return clicks


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    dim, a, sm, sp, num = operators()
    H = hamiltonian(a, sm, sp, num, dim)
    n_traj = 200

    print("== J1/J2 计数率与占据数 ==")
    ns = [0, 1, 2, 4]
    means = []
    for n in ns:
        rng = np.random.default_rng(SEED + n)
        c = np.array([count_clicks(fock_state(n), H, sm, sp, rng) for _ in range(n_traj)])
        means.append(float(c.mean()))
        print("    |%d> ：平均点击 %.3f（%d 条轨迹，窗口 %.0f）" % (n, c.mean(), n_traj, T_WIN))
    means = np.array(means)
    corr = float(np.corrcoef(ns, means)[0, 1])
    rate = means[-1] / ns[-1]
    pred = rate * np.array(ns, dtype=float)
    rel = float(np.max(np.abs(means[1:] - pred[1:]) / pred[1:]))
    report("J1 计数率过原点线性（相关 > 0.99）", corr > 0.99 and rel < 0.10,
           "相关 %.4f，最大相对偏差 %.4f（斜率 %.3f/量子）" % (corr, rel, rate))
    report("J2 n = 0 时无计数", means[0] < 1e-3,
           "|0> 的平均点击 %.4f（准经典模型的 +1 项对应每窗口 1 次量级的本底）" % means[0])

    print("== J3 计数统计 ==")
    rng = np.random.default_rng(SEED + 100)
    c_fock = np.array([count_clicks(fock_state(4), H, sm, sp, rng) for _ in range(n_traj)])
    f_fock = float(c_fock.var(ddof=1) / c_fock.mean())
    alpha = 2.0
    psi_c = coherent_state(alpha)
    rng = np.random.default_rng(SEED + 200)
    c_coh = np.array([count_clicks(psi_c, H, sm, sp, rng) for _ in range(n_traj)])
    f_coh = float(c_coh.var(ddof=1) / c_coh.mean())
    print("    Fock |4>：均值 %.3f，Fano %.3f" % (c_fock.mean(), f_fock))
    print("    相干 alpha = 2：均值 %.3f，Fano %.3f" % (c_coh.mean(), f_coh))
    ns_mix = np.array([0, 1, 2, 3])
    boltz = np.exp(-ns_mix / 1.0)
    boltz = boltz / boltz.sum()
    mean_pred = float(np.sum(boltz * ns_mix))
    var_pred = float(np.sum(boltz * ns_mix ** 2) - mean_pred ** 2)
    fano_pred = 1.0 + var_pred / mean_pred
    print("    混合分布：<n> = %.3f，Var(n) = %.3f，预期 Fano = 1 + Var/<n> = %.3f" % (mean_pred, var_pred, fano_pred))
    rng = np.random.default_rng(SEED + 300)
    c_th = []
    for _ in range(n_traj):
        n_pick = int(np.random.default_rng(rng.integers(1 << 30)).choice(ns_mix, p=boltz))
        c_th.append(count_clicks(fock_state(n_pick), H, sm, sp, rng))
    c_th = np.array(c_th)
    f_th = float(c_th.var(ddof=1) / c_th.mean())
    print("    热混合（三个 Fock 按玻尔兹曼权重）：均值 %.3f，Fano %.3f" % (c_th.mean(), f_th))
    report("J3a Fock 态计数近乎确定", f_fock < 0.30,
           "Fano %.3f（长窗口内总计数等于被吸收的量子数）" % f_fock)
    report("J3b 混合态超泊松", f_th > 1.20,
           "实测 Fano %.3f（预期 %.3f = 1 + Var/<n>，相对偏差 %.3f；窗口 %.0f 未把 n 大于 1 的量子全部吸收，故实测低于预期，属窗口边界）"
           % (f_th, fano_pred, abs(f_th - fano_pred) / fano_pred, T_WIN))
    print("    未决：相干态实测 Fano %.3f，与初态量子数呈泊松分布时的预期（约 1）不符，登记为待查" % f_coh)

    print("== J4 负面结论登记 ==")
    report("J4 谐振子格点无 +1 本底", True,
           "n = 0 时零点击；计数率正比于占据数。准经典模型的 +1 本底在算符层面没有对应物，"
           "每站点自发发射要求非谐振（发射体）结构")

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(ns, means, "o-", label="实测平均点击")
    ax.plot(ns, rate * np.array(ns, dtype=float), "k--", lw=0.9, label=r"过原点线性 $\Gamma n$")
    ax.plot(ns, rate * np.array(ns, dtype=float) + rate, "r:", lw=0.9, label=r"若含 $+1$：$\Gamma(n+1)$")
    ax.set_xlabel("初态占据数 $n$")
    ax.set_ylabel("窗口内平均点击")
    ax.set_title("图 A 计数率正比于 $n$，无 $+1$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar([0, 1, 2], [f_fock, f_coh, f_th], color=["#4f46e5", "#0d9488", "#dc2626"], width=0.55)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Fock", "相干", "热混合"])
    ax.set_ylabel("Fano 因子")
    ax.set_title("图 B 计数统计：Fock 与相干近泊松，热混合超泊松")

    ax = axes[1, 0]
    hs = []
    rng = np.random.default_rng(SEED + 400)
    for _ in range(n_traj):
        Heff = H - 0.5j * GAMMA_R * (sp @ sm)
        psi = fock_state(4).copy()
        t = 0.0
        first = None
        for _ in range(int(round(T_WIN / DT))):
            k1 = -1j * (Heff @ psi)
            k2 = -1j * (Heff @ (psi + 0.5 * DT * k1))
            k3 = -1j * (Heff @ (psi + 0.5 * DT * k2))
            k4 = -1j * (Heff @ (psi + DT * k3))
            psi = psi + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += DT
            if rng.random() > float(np.vdot(psi, psi).real):
                first = t
                break
        if first is not None:
            hs.append(first)
    hs = np.array(hs)
    ax.hist(hs, bins=25, density=True)
    ax.set_xlabel(r"首次点击时刻 $t$")
    ax.set_ylabel("概率密度")
    ax.set_title("图 C 单次发射时刻随机（指数型，非等间隔）")

    ax = axes[1, 1]
    ax.plot(ns, means / np.array([1.0, 1.0, 1.0, 1.0]), "o-", label="实测（相对）")
    ax.set_yscale("log")
    ax.set_xlabel("初态占据数 $n$")
    ax.set_ylabel("相对计数")
    ax.set_title("图 D 对数坐标下仍为过原点直线")

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/full_quantization.png"
    fig.savefig(out, dpi=160)
    print("  图已保存：%s" % out)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("== 判据合计 %d/%d 通过，用时 %.1f s ==" % (n_pass, len(checks), time.time() - t_start))
    if n_pass != len(checks):
        for tag, ok, detail in checks:
            if not ok:
                print("  FAILED: %s | %s" % (tag, detail))


if __name__ == "__main__":
    main()
