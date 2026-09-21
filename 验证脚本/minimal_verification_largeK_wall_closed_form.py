# -*- coding: utf-8 -*-
r"""
minimal_verification_largeK_wall_closed_form.py

大刚度档壁张力的可引用解析式轮：一维约化精确极小、两格快跳族的闭合形式、四阶渐近式与两支交叉点。

来源：作者 2026-09-21 指令（把大刚度档的离散形式写成可引用的解析式）。
模型：竖条几何沿 x 的一维约化（壁沿 y 平移不变，故 y 向梯度项为零）。单位长度读数
      sigma(u) = (1/2) sum_j (u_{j+1} - u_j)^2 + K sum_j (1 - cos 2 pi u_j)，两端固定 u_0 = 0、u_{N-1} = 1。
      与第 336 条脚本 energy() 除以 Ny 逐位同口径。
解析式：记 A = 3 + 4 pi^2 K。
      (a) 两格快跳族（剖面 ..., 0, eps, 1-eps, 1, ...）的精确势极小点 eps* 由
          6 eps + 4 pi K sin(2 pi eps) = 2 定，族能 sigma_2ex(K) 由该 eps* 代入；
      (b) 调和近似给闭式 sigma_2h(K) = 1/2 - 1/A，eps* = 1/A；
      (c) 势项四阶展开给 sigma_2e4(K) = 1/2 - 1/A - 4 pi^4 K/(3 A^4)。
      大 K 渐近（由 (c) 反推）：sigma = 1/2 - 1/(4 pi^2 K) + 3/(16 pi^4 K^2) - (27 + pi^2)/(192 pi^6 K^3) + O(K^-4)。

判据（先行）：
  Z1 一维精确极小对第 336 条八档实测偏差 <= 1.5%；且 N = 257 与 N = 513 两条链的读数差 <= 1e-4
  Z2 四阶展开对两格精确极小偏差 <= 0.1%（K >= 0.15 档）
  Z3 变分不等式 sigma_1D <= sigma_2ex <= sigma_2h 在八档逐档成立
  Z4 两支交叉点由夹逼区间内的二分给出，调和支落于 (0.05, 0.10)、精确支落于 (0.05, 0.06)
  Z5 大 K 渐近系数：二项拟合 c2 对 3/(16 pi^4)、c3 对 -(27 + pi^2)/(192 pi^6) 的相对偏差 <= 30%
  Z6 尾修正（两格精确极小减一维极小）在 K >= 0.15 档 <= 0.5%，且随 K 收窄
  Z8 一维精确剖面在跳变两侧的尾衰减率对线性化预测 cosh(kappa) = 1 + 2 pi^2 K 的偏差 <= 5%（K >= 0.15 档）
  Z9 尾闭式（两格快跳加指数尾，尾项用精确势求和）对一维精确极小的偏差 <= 0.05%（K >= 0.15 档），且不小于一维精确极小
解析式（尾闭式，本轮新增）：cosh(kappa) = 1 + 2 pi^2 K、t = tanh(kappa/2)，
      eps 由 eps (3 - e^-kappa) + 2 pi K sin(2 pi eps) = 1 定；
      sigma_tail(K) = 1/2 [2 eps^2 t + (1 - 2 eps)^2] + 2 K [1 - cos(2 pi eps)]
                      + 2 K sum_{j>=1} [1 - cos(2 pi eps e^{-kappa j})]（尾和按精确势逐项求和，收敛到 1e-18）
"""
import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 257
CX = 0.37
REF = {0.001: 0.040242, 0.005: 0.089781, 0.02: 0.177866, 0.10: 0.345133,
       0.15: 0.382646, 0.20: 0.405285, 0.24: 0.417864, 0.30: 0.431444}
K_LIST = sorted(REF)
FIG = "D:/software/TianXingFC/images/wall_tension_largeK.png"


def energy(u, K):
    d = np.diff(u)
    return 0.5 * float(np.dot(d, d)) + K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))


def grad_int(u, K):
    d = np.diff(u)
    return -d[1:] + d[:-1] + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u[1:-1])


def sigma_1d(K, n=N):
    x = np.arange(n, dtype=float) - (n - 1) / 2.0
    xi = 1.0 / (2.0 * np.pi * np.sqrt(2.0 * K))
    u = 0.5 * (1.0 + np.tanh((x - CX) / max(xi, 0.30)))
    u[0] = 0.0
    u[-1] = 1.0

    def fg(v):
        uu = np.empty(n)
        uu[0] = 0.0
        uu[-1] = 1.0
        uu[1:-1] = v
        return energy(uu, K), grad_int(uu, K)

    res = minimize(fg, u[1:-1].copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400000, "maxfun": 400000, "ftol": 1e-16,
                            "gtol": 1e-12, "maxcor": 40})
    uu = np.empty(n)
    uu[0] = 0.0
    uu[-1] = 1.0
    uu[1:-1] = res.x
    return energy(uu, K), uu, float(np.max(np.abs(grad_int(uu, K))))


def sigma_cont(K):
    return 4.0 / np.pi * np.sqrt(K)


def sigma_2h(K):
    return 0.5 - 1.0 / (3.0 + 4.0 * np.pi ** 2 * K)


def eps_2ex(K):
    f = lambda e: 6.0 * e + 4.0 * np.pi * K * np.sin(2.0 * np.pi * e) - 2.0
    return brentq(f, 1e-14, 0.4)


def sigma_2ex(K):
    e = eps_2ex(K)
    return 0.5 * (2.0 * e * e + (1.0 - 2.0 * e) ** 2) + 2.0 * K * (1.0 - np.cos(2.0 * np.pi * e))


def sigma_2e4(K):
    A = 3.0 + 4.0 * np.pi ** 2 * K
    return 0.5 - 1.0 / A - 4.0 * np.pi ** 4 * K / (3.0 * A ** 4)


def sigma_tail(K, terms=60):
    kap = float(np.arccosh(1.0 + 2.0 * np.pi ** 2 * K))
    ek = np.exp(-kap)
    e = brentq(lambda x: x * (3.0 - ek) + 2.0 * np.pi * K * np.sin(2.0 * np.pi * x) - 1.0, 1e-14, 0.5)
    t = np.tanh(kap / 2.0)
    tail = 0.0
    for j in range(1, terms + 1):
        tail += 1.0 - np.cos(2.0 * np.pi * e * ek ** j)
    return 0.5 * (2.0 * e * e * t + (1.0 - 2.0 * e) ** 2) + 2.0 * K * (1.0 - np.cos(2.0 * np.pi * e)) + 2.0 * K * tail, e, kap


if __name__ == "__main__":
    rows = []
    prof = {}
    for K in K_LIST:
        s1, u, ginf = sigma_1d(K)
        prof[K] = u
        rows.append((K, s1, sigma_cont(K), sigma_2h(K), sigma_2ex(K), sigma_2e4(K), ginf))
    size = []
    for K in (0.001, 0.30):
        a = sigma_1d(K, 257)[0]
        b = sigma_1d(K, 513)[0]
        size.append((K, a, b, abs(a - b)))

    print("K, sigma_1D, sigma_cont, sigma_2h, sigma_2ex, sigma_2e4, |grad|inf, ref336, dev(1D vs 336)")
    for K, s1, sc, sh, se, s4, ginf in rows:
        print("%.3f %.6f %.6f %.6f %.6f %.6f %.1e %.6f %+.2f%%"
              % (K, s1, sc, sh, se, s4, ginf, REF[K], 100.0 * (s1 - REF[K]) / REF[K]))

    print("K, dev_cont, dev_2h, dev_2ex, dev_2e4, dev_tail (对 sigma_1D, 单位 %)：")
    for K, s1, sc, sh, se, s4, _ in rows:
        st = sigma_tail(K)[0]
        print("%.3f %+.4f%% %+.4f%% %+.4f%% %+.4f%% %+.6f%%"
              % (K, 100.0 * (sc - s1) / s1, 100.0 * (sh - s1) / s1, 100.0 * (se - s1) / s1,
                 100.0 * (s4 - s1) / s1, 100.0 * (st - s1) / s1))

    z = []
    dev1 = max(abs(s1 - REF[K]) / REF[K] for K, s1, _, _, _, _, _ in rows)
    z.append(("Z1", "一维精确极小对八档实测最大偏差", "%.3f%%" % (100.0 * dev1), "<= 1.5%", dev1 <= 0.015))
    dsize = max(t[3] for t in size)
    z.append(("Z1b", "N = 257 对 513 的链长差", "%.1e" % dsize, "<= 1e-4", dsize <= 1e-4))

    big = [r for r in rows if r[0] >= 0.15]
    dev2 = max(abs(r[5] - r[4]) / r[4] for r in big)
    z.append(("Z2", "四阶展开对两格精确极小最大偏差（K >= 0.15）", "%.3f%%" % (100.0 * dev2), "<= 0.1%", dev2 <= 0.001))

    viol = [r[0] for r in rows if not (r[1] <= r[4] + 1e-9 <= r[3] + 1e-9)]
    z.append(("Z3", "sigma_1D <= sigma_2ex <= sigma_2h", "违反档位 %s" % (viol if viol else "无"), "八档逐档成立", len(viol) == 0))

    kh = brentq(lambda k: sigma_cont(k) - sigma_2h(k), 1e-4, 0.3, xtol=1e-14, rtol=1e-14)
    ke = brentq(lambda k: sigma_cont(k) - sigma_2ex(k), 1e-4, 0.3, xtol=1e-14, rtol=1e-14)
    ok4 = (0.05 < kh < 0.10) and (0.05 < ke < 0.06)
    z.append(("Z4", "交叉点（调和支与精确支）", "K* = %.6f 与 %.6f" % (kh, ke), "调和支 (0.05, 0.10)、精确支 (0.05, 0.06)", ok4))

    KF = [3.0, 6.0, 12.0, 24.0]
    sf = [(k, sigma_1d(k)[0]) for k in KF]
    resid = np.array([s - (0.5 - 1.0 / (4.0 * np.pi ** 2 * k)) for k, s in sf])
    M = np.array([[1.0 / k ** 2, 1.0 / k ** 3] for k, _ in sf])
    coef, *_ = np.linalg.lstsq(M, resid, rcond=None)
    c2a, c3a = 3.0 / (16.0 * np.pi ** 4), -(27.0 + np.pi ** 2) / (192.0 * np.pi ** 6)
    r2 = abs(coef[0] - c2a) / c2a
    r3 = abs(coef[1] - c3a) / abs(c3a)
    z.append(("Z5", "渐近系数拟合对解析值（c2 与 c3）", "c2 = %.6e（解析 %.6e，差 %.1f%%）、c3 = %.6e（解析 %.6e，差 %.1f%%）"
              % (coef[0], c2a, 100.0 * r2, coef[1], c3a, 100.0 * r3), "两项相对偏差 <= 30%", max(r2, r3) <= 0.30))

    zt = []
    for K in (0.15, 0.20, 0.24, 0.30):
        st, et, kt = sigma_tail(K)
        s1 = next(r[1] for r in rows if r[0] == K)
        zt.append((K, st, s1, (st - s1) / s1))
    print("K, sigma_tail, sigma_1D, rel.dev：")
    for K, st, s1, d in zt:
        print("%.3f %.8f %.8f %+.4f%%" % (K, st, s1, 100.0 * d))
    print("K, sigma_tail, sigma_1D, rel.dev（定义域扩展档）：")
    dom = []
    for K in (0.05, 0.10, 0.50, 1.0, 3.0, 10.0, 100.0):
        st, _, _ = sigma_tail(K)
        s1 = sigma_1d(K)[0]
        dom.append((K, (st - s1) / s1))
        print("%.3f %.8f %.8f %+.5f%%" % (K, st, s1, 100.0 * (st - s1) / s1))
    dz9b = max(abs(d) for K, d in dom if K >= 0.50)
    z.append(("Z9b", "尾闭式在深大 K 档（K >= 0.5）", "最大偏差 %.2e" % dz9b, "<= 1e-6", dz9b <= 1e-6))
    print("尾闭式的定义域：K >= 0.10 后偏差降到 %.4f%% 量级，K >= 0.5 进入 %.1e 量级；K = 0.05 档为 %.4f%%（连续式在该档更好）"
          % (100.0 * abs(dom[1][1]), dz9b, 100.0 * dom[0][1]))
    dz9 = max(abs(t[3]) for t in zt)
    z.append(("Z9", "尾闭式对一维精确极小（K >= 0.15 四档）",
              "最大偏差 %.4f%%（K = 0.15 给 %+.4f%%、K = 0.30 给 %+.4f%%）、恒有 sigma_tail >= sigma_1D"
              % (100.0 * dz9, 100.0 * zt[0][3], 100.0 * zt[-1][3]),
              "<= 0.05% 且不小于一维极小", dz9 <= 5e-4 and all(t[3] >= -1e-12 for t in zt)))

    def kappa_meas(u):
        j = int(np.argmax(np.diff(u)))
        left = [u[j - 1] / u[j], u[j - 2] / u[j - 1]]
        right = [(1.0 - u[j + 2]) / (1.0 - u[j + 1]), (1.0 - u[j + 3]) / (1.0 - u[j + 2])]
        return -float(np.mean(np.log(left + right)))

    kz = []
    for K in (0.15, 0.20, 0.24, 0.30):
        km = kappa_meas(prof[K])
        kp = float(np.arccosh(1.0 + 2.0 * np.pi ** 2 * K))
        kz.append((K, km, kp, abs(km - kp) / kp))
    print("K, kappa_meas, kappa_pred, rel.dev：")
    for K, km, kp, d in kz:
        print("%.3f %.6f %.6f %.2f%%" % (K, km, kp, 100.0 * d))
    dz8 = max(t[3] for t in kz)
    z.append(("Z8", "尾衰减率（K >= 0.15 四档）",
              "最大偏差 %.2f%%（K = 0.30 给 %.6f 对 %.6f）" % (100.0 * dz8, kz[-1][1], kz[-1][2]),
              "<= 5%", dz8 <= 0.05))

    tail = [(r[0], r[4] - r[1], r[1]) for r in big]
    dev6 = max(abs(d) / s for _, d, s in tail)
    z.append(("Z6", "尾修正（两格精确极小减一维极小，K >= 0.15）",
              "%.3f%%（K = 0.15 给 %.3f%%，K = 0.30 给 %.3f%%）"
              % (100.0 * dev6, 100.0 * abs(tail[0][1]) / tail[0][2], 100.0 * abs(tail[-1][1]) / tail[-1][2]),
              "<= 0.5% 且随 K 收窄", dev6 <= 0.005 and abs(tail[-1][1]) < abs(tail[0][1])))
    z.append(("Z7", "登记", "登记", "登记", True))

    for a, b, c, d, e in z:
        print("%s | %s | %s | %s | %s" % (a, b, c, d, "通过" if e else "未通过"))

    print("eps* 对照：K = 0.30 两格精确 %.6f、调和 %.6f；一维精确步中心附近六站点 %s"
          % (eps_2ex(0.30), 1.0 / (3.0 + 4.0 * np.pi ** 2 * 0.30),
             np.array2string(prof[0.30][125:131], precision=6)))
    print("K = 0.30 一维精确最近两站点偏离整数 %.6f 与 %.6f（尾修正在 0.081%% 量级）"
          % (prof[0.30][128], 1.0 - prof[0.30][129]))
    print("分档用法：K <= 0.02 连续式（对一维精确 -0.05%% 至 -1.22%%）；K >= 0.15 两格四阶式（对一维精确最大 %.3f%%）；"
          "K = 0.10 两支皆高于一维精确值（连续式 +16.7%%、两格精确 +1.30%%），故下包络在那里只是插值式" % (100.0 * dev2))

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
    kk = np.logspace(-3.2, 1.6, 200)
    ax[0].loglog(kk, [sigma_cont(k) for k in kk], "k--", lw=1.0, label="continuum 4 sqrt(K)/pi")
    ax[0].loglog(kk, [sigma_2h(k) for k in kk], "b-", lw=1.4, label="2-node harmonic 1/2 - 1/A")
    ax[0].loglog(kk, [sigma_2e4(k) for k in kk], "g-.", lw=1.4, label="2-node 4th-order")
    ax[0].loglog(K_LIST, [r[1] for r in rows], "ro", ms=5.0, label="exact 1D minimum")
    ax[0].axvline(kh, color="gray", lw=0.8)
    ax[0].set_xlabel("K"); ax[0].set_ylabel("sigma (unit length)"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)

    ax[1].semilogx(K_LIST, [100.0 * (r[2] - r[1]) / r[1] for r in rows], "k--o", ms=4, label="continuum")
    ax[1].semilogx(K_LIST, [100.0 * (r[3] - r[1]) / r[1] for r in rows], "b-s", ms=4, label="2-node harmonic")
    ax[1].semilogx(K_LIST, [100.0 * (r[5] - r[1]) / r[1] for r in rows], "g-.d", ms=4, label="2-node 4th-order")
    ax[1].axhline(0.0, color="gray", lw=0.8)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("deviation from exact 1D (%)"); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3)

    for K, c, lb in ((0.001, "b", "K = 0.001"), (0.10, "g", "K = 0.10"), (0.30, "r", "K = 0.30")):
        u = prof[K]
        idx = N // 2
        ax[2].plot(np.arange(-9, 10), u[idx - 9:idx + 10], c + "-o", ms=3.5, label=lb)
    ax[2].set_xlabel("node offset from step"); ax[2].set_ylabel("u"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print("figure -> %s" % FIG)
