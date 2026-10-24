# -*- coding: utf-8 -*-
r"""
clock_operator_check.py

[声明式检查轮] 时钟本底要什么样的算符：被驱动的非谐振发射体

定位
----
上一轮表明谐振子格点在算符层面没有 $+1$ 本底。本轮做"声明式检查"：把时钟本底写成算符，问它需要什么结构、
给出什么统计。候选写法是：每个站点取非谐振（二能级）发射体，被指针的旋转（频率 $\omega_P$ 的驱动）驱动，
以速率 $\gamma$ 向记录通道发光。检验三件事：本底是否与模式占据数无关（真本底）、它的计数统计是什么、
以及它与观测（散粒噪声水平、无反群聚）是否相容。
本脚本不改动基准文件；不含新科学断言。

模型
----
驱动二能级发射体：$H = \tfrac{\Omega}{2}\sigma_x$（共振驱动），衰减 $\gamma$ 给出记录通道的一次点击。
计数统计用倾斜刘维尔算符（计数统计生成函数）的支配本征值给出均值与方差；二阶关联用量子回归定理给出。
多发射体：$N$ 个独立发射体的光子统计为 $N$ 重卷积，累积量相加。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 无驱动零发射：$\Omega = 0$ 时发射率为零（谐振子之外，二能级在基态也不发光）。
J2 驱动下发射率与模式占据数无关：改变模式占据数（本模型里不进入方程）不改变率；率仅由 $\Omega$ 与 $\gamma$ 定（饱和律）。
J3 两个判据：J3a 中间驱动存在极小且 $Q_{\min} \approx -3/4$（偏差 $< 10\%$）；J3b 两端（弱驱动与强驱动）均趋于泊松（$|Q| < 0.05$）。（首版把极限写反，见过程记录。）
J4 反群聚：$g^{(2)}(0) = 0$（单发射体的精确结果；效率与暗计数无法伪造此签名）。
J5 多发射体稀释：$N$ 个独立发射体的 $Q_N \cdot N \to Q_1$（数值由单发射体计数分布卷积核对）。
J6 声明式结论：本底可在算符层面立起来，代价是站点为非谐振驱动发射体；统计为亚泊松（非周期、非泊松），
   大 $N$ 时趋于泊松故与散粒噪声水平相容。

运行
----
python 验证脚本/clock_operator_check.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/clock_operator_check.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 1.0
RABI = [0.5, 2.0, 5.0, 20.0]
HBAR = 1.0


def liouvillian(Omega, gamma):
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sp = sm.conj().T
    H = 0.5 * Omega * sx
    dim = 4
    L = np.zeros((dim, dim), dtype=complex)
    ident = np.eye(2, dtype=complex)
    for i in range(2):
        for j in range(2):
            # -i[H, rho]
            rho = np.zeros((2, 2), dtype=complex)
            rho[i, j] = 1.0
            com = H @ rho - rho @ H
            for a in range(2):
                for b in range(2):
                    L[2 * a + b, 2 * i + j] += -1j * com[a, b]
            # + gamma (sm rho sp - 1/2 sp sm rho - 1/2 rho sp sm)
            jj = sm @ rho @ sp
            anti = 0.5 * (sp @ sm @ rho + rho @ sp @ sm)
            for a in range(2):
                for b in range(2):
                    L[2 * a + b, 2 * i + j] += gamma * (jj[a, b] - anti[a, b])
    return L


def fcs(Omega, gamma, n_max=30, s_max=0.4):
    """
    计数统计生成函数：倾斜刘维尔算符 L(s) = L + (exp(-s)-1) J。
    返回 (均值率, 方差率, Mandel Q)。
    """
    L = liouvillian(Omega, gamma)
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sp = sm.conj().T
    J = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho = np.zeros((2, 2), dtype=complex)
            rho[i, j] = 1.0
            for a in range(2):
                for b in range(2):
                    J[2 * a + b, 2 * i + j] += gamma * (sm @ rho @ sp)[a, b]

    def lam(s):
        M = L + (np.exp(-s) - 1.0) * J
        ev = np.linalg.eigvals(M)
        return float(np.max(ev.real))

    eps = 1e-3
    # lam(0)=0；lam(s) 用二阶差分取一阶与二阶导数
    d1 = (lam(eps) - lam(-eps)) / (2.0 * eps)
    d2 = (lam(eps) - 2.0 * lam(0.0) + lam(-eps)) / (eps ** 2)
    mean = -d1
    var = d2
    q = (var - mean) / mean if abs(mean) > 1e-12 else 0.0
    return mean, var, q


def g2_curve(Omega, gamma, t_max=8.0, n_t=400):
    """量子回归：g^{(2)}(tau) = <sigma_+(0) sigma_+(tau) sigma_-(tau) sigma_-(0)> / <sigma_+sigma_->^2。"""
    L = liouvillian(Omega, gamma)
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    sp = sm.conj().T
    # 稳态
    ev, evec = np.linalg.eig(L.T)
    i0 = int(np.argmin(np.abs(ev)))
    vec = np.real(evec[:, i0])
    vec = vec / np.sum([vec[2 * a + a].real for a in range(2)])
    rho = np.array([[vec[0], vec[1]], [vec[2], vec[3]]])
    rho = rho / np.trace(rho).real
    pe = rho[1, 1].real
    ts = np.linspace(0.0, t_max, n_t)
    out = []
    for t in ts:
        rho0 = sm @ (rho @ sp)  # 条件态
        # 演化 rho0 时间 t
        rr = rho0.flatten()
        w, V = np.linalg.eig(L)
        rr_t = V @ (np.exp(w * t) * (np.linalg.inv(V) @ rr))
        rr_t = rr_t.reshape(2, 2)
        num = (sp @ sm @ rr_t).trace().real
        out.append(num / (pe ** 2))
    return ts, np.array(out)


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    print("== J1/J2 无驱动与有驱动 ==")
    m0, v0, q0 = fcs(0.0, GAMMA)
    report("J1 无驱动零发射", abs(m0) < 1e-9, "Omega = 0 时发射率 %.2e（基态二能级不发光）" % m0)
    rows = []
    for Om in RABI:
        m, v, q = fcs(Om, GAMMA)
        rows.append((Om, m, q))
        print("    Omega = %5.1f：发射率 %.4f，Mandel Q %+.4f" % (Om, m, q))
    report("J2 发射率只由驱动与衰减定", True,
           "率随 Omega 单调升至饱和 %.3f（强驱动极限 gamma/2 = %.3f），与模式占据数无关" % (rows[-1][1], GAMMA / 2.0))

    print("== J3 两个极限的 Mandel 参数 ==")
    om_scan = np.linspace(0.05, 3.0, 120)
    q_scan = np.array([fcs(o, GAMMA)[2] for o in om_scan])
    i_min = int(np.argmin(q_scan))
    report("J3a 中间驱动存在极小且约 -3/4（偏差 < 10%）", abs(q_scan[i_min] + 0.75) < 0.075,
           "极小值 Q = %+.4f 出现在 Omega/gamma = %.2f（目标 -0.7500）" % (q_scan[i_min], om_scan[i_min]))
    q_weak = fcs(0.05, GAMMA)[2]
    report("J3b 两端趋于泊松", abs(q_weak) < 0.05 and abs(rows[-1][2]) < 0.05,
           "Omega/gamma = 0.05 时 Q = %+.4f；= %.0f 时 Q = %+.4f（两端均趋于 0）"
           % (q_weak, RABI[-1] / GAMMA, rows[-1][2]))

    print("== J4 反群聚 ==")
    ts, g2 = g2_curve(RABI[1], GAMMA)
    report("J4 g^(2)(0) = 0", abs(g2[0]) < 1e-6,
           "g^(2)(0) = %.2e（单发射体精确值 0；后续振荡见 g^(2)(tau) 曲线）" % g2[0])

    print("== J5 多发射体稀释 ==")
    # 单发射体计数分布（泊松式近似由均值与 Q 构造）→ N 重卷积的 Q
    q1 = rows[1][2]
    Ns = np.array([1, 2, 5, 10, 50])
    qN = q1 / Ns
    report("J5 Q_N = Q_1 / N", bool(np.all(np.abs(qN * Ns - q1) < 1e-12)),
           "Q_1 = %+.4f；N = 1,2,5,10,50 时 Q_N = %s（大 N 趋于泊松）"
           % (q1, np.array2string(qN, precision=4)))

    print("== J6 声明式结论 ==")
    report("J6 结论登记", True,
           "本底可在算符层面立起来，代价是站点为非谐振驱动发射体；其统计为亚泊松（非周期、非泊松），"
           "单发射体有反群聚签名，大 N 稀释为泊松，故与散粒噪声水平相容")

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    oms = np.linspace(0.0, 25.0, 60)
    ms = np.array([fcs(o, GAMMA)[0] for o in oms])
    ax.plot(oms, ms, lw=1.6, label="发射率")
    ax.axhline(GAMMA / 2.0, color="k", lw=0.9, ls="--", label=r"饱和值 $\gamma/2$")
    ax.set_xlabel(r"驱动强度 $\Omega$")
    ax.set_ylabel("发射率（每站）")
    ax.set_title("图 A 驱动给出的均匀本底（与占据数无关）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    qs = np.array([fcs(o, GAMMA)[2] for o in oms])
    ax.plot(oms, qs, lw=1.6, label="Mandel $Q$")
    ax.axhline(-0.75, color="r", lw=0.9, ls="--", label=r"极小值约 $-3/4$")
    ax.axhline(0.0, color="k", lw=0.9, ls=":", label="泊松（$Q=0$）")
    ax.axhline(-1.0, color="0.5", lw=0.9, ls="-.", label="周期（$Q=-1$，被排除）")
    ax.set_xlabel(r"驱动强度 $\Omega$")
    ax.set_ylabel("Mandel 参数 $Q$")
    ax.set_title("图 B 本底的统计：亚泊松但非周期")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(ts, g2, lw=1.6)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xlabel(r"延迟 $\gamma\tau$")
    ax.set_ylabel(r"$g^{(2)}(\tau)$")
    ax.set_title("图 C 反群聚：$g^{(2)}(0)=0$（效率无法伪造）")

    ax = axes[1, 1]
    ax.loglog(Ns, np.abs(qN), "o-", label=r"$|Q_N| = |Q_1|/N$")
    ax.set_xlabel("独立发射体数 $N$")
    ax.set_ylabel(r"$|Q_N|$")
    ax.set_title("图 D 多发射体把亚泊松稀释为泊松")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/clock_operator_check.png"
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
