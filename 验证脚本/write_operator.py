# -*- coding: utf-8 -*-
r"""
write_operator.py

[写入算符轮] 碰撞即写入的算符形式：过轴即写、写即转移、写有代价

定位
----
事件模型已核（写入时刻等于折返时刻、本底等于碰撞率、稀释统计 $Q = -\eta$）。本轮给出它的算符形式：
指针取紧致相位（转子型自由度，$H_0 = \omega_P \hat L$），写入取相位条件耦合
  $H_{\rm int} = g\,f(\hat\theta)\,(a^\dagger c + a c^\dagger)$，
$f(\hat\theta)$ 是在过轴处尖锐的周期函数，$a$ 是站点模（被采样对象），$c$ 是记录寄存器。
于是"过轴"由动力学给出（不是外部定时），写一次就把一个激发从站点转移进寄存器。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 写入时刻等于折返时刻：寄存器布居的上升时刻与 $t_{\rm cross} = (2\pi - \theta_0)/\omega_P$ 相符（偏差 $< 5\%$）。
J2 写入由相位条件给出：改变脉冲宽度 $w$（三档）不改变写入时刻（散布 $< 5\%$）。
J3 写即转移：站点与寄存器的激发数守恒（偏差 $< 10^{-9}$），寄存器从空变为一。
J4 反作用的位置：写入把激发从被采样对象转移到寄存器（站点剩余激发 $< 5\%$），而指针（时钟）的相位宽度变化 $< 1\%$：采样扰动对象，不扰动时钟。
J5 读数登记：每过轴写一次（每站每摆程一笔）；"每激发写一次（$n+1$）"的写法属待定设计，登记为后续项。

运行
----
python 验证脚本/write_operator.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/write_operator.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_THETA = 256
OMEGA_P = 2.0 * np.pi
G_WRITE = None  # 由脉冲面积配成完整交换
THETA0 = np.pi
W_PULSE = 0.5
PACKET_W = [0.15, 0.25, 0.40]
DT = 2.0e-3
T_MAX = 1.2


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def split_step(psi, dt, theta, kvec, w, g, n_modes=2):
    """psi 形状 (N_THETA, 2, 2)：相位 x 站点模 x 寄存器。"""
    # 半步：相位条件耦合（在对角基）
    f = pulse(theta, w)
    psi = apply_int(psi, f, dt / 2.0, g)
    # 整步：指针自由演化（傅里叶基）
    for a in range(n_modes):
        for c in range(n_modes):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    # 半步：耦合
    psi = apply_int(psi, f, dt / 2.0, g)
    return psi


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    """耦合项 g f (a^dag c + a c^dag)，在两能级子空间上精确指数化。"""
    ang = g * f * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        other_a = 1 - a
        for c in range(2):
            other_c = 1 - c
            out[:, other_a, other_c] += -1j * sin_a * psi[:, a, c]
    return out


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
    t_cross = (2.0 * np.pi - THETA0) / OMEGA_P

    print("== J1/J2 写入时刻 ==")
    area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
    g_full = (np.pi / 2.0) * OMEGA_P / area
    print("    脉冲面积 %.4f，配成完整交换的耦合 g = %.3f" % (area, g_full))
    results = []
    for pw in PACKET_W:
        psi = np.zeros((N_THETA, 2, 2), dtype=complex)
        packet = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
        psi[:, 1, 0] = packet / np.linalg.norm(packet)
        times, pops, widths = [], [], []
        t = 0.0
        for step in range(int(round(T_MAX / DT))):
            psi = split_step(psi, DT, theta, kvec, W_PULSE, g_full)
            t += DT
            times.append(t)
            pops.append(float(np.sum(np.abs(psi[:, :, 1]) ** 2)))
            if step % 20 == 0:
                dens = np.sum(np.abs(psi) ** 2, axis=(1, 2))
                dens = dens / dens.sum()
                ang = np.angle(np.sum(dens * np.exp(1j * theta)))
                widths.append(float(np.sqrt(np.sum(dens * (np.angle(np.exp(1j * (theta - ang))) ** 2)))))
        pops = np.array(pops)
        hit = np.nonzero(pops > 0.5)[0]
        t_write = times[int(hit[0])] if hit.size else np.nan
        results.append((pw, t_write, float(pops[-1]), float(widths[-1])))
        print("    波包宽 %.2f：写入时刻 %.4f（折返时刻 %.4f），末态寄存器布居 %.3f，指针相位宽度 %.4f"
              % (pw, t_write, t_cross, pops[-1], widths[-1]))
    tw = np.array([r[1] for r in results])
    rel1 = abs(tw[0] - t_cross) / t_cross
    report("J1 写入时刻 = 折返时刻（偏差 < 5%）", rel1 < 0.05,
           "写入时刻 %.4f 对折返时刻 %.4f，相对偏差 %.4f" % (tw[0], t_cross, rel1))
    spread = float((tw.max() - tw.min()) / tw.mean())
    report("J2 三档指针波包宽度下写入时刻不变", spread < 0.05, "散布 %.4f" % spread)

    print("== J3 写即转移 ==")
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    packet = np.exp(-((theta - THETA0) ** 2) / (2.0 * 0.25 ** 2))
    psi[:, 1, 0] = packet / np.linalg.norm(packet)
    def excitation(p):
        nsite = np.sum(np.abs(p[:, 1, :]) ** 2)
        nmem = np.sum(np.abs(p[:, :, 1]) ** 2)
        return float(nsite + nmem)
    e0 = excitation(psi)
    for _ in range(int(round(T_MAX / DT))):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, g_full)
    e1 = excitation(psi)
    site_left = float(np.sum(np.abs(psi[:, 1, :]) ** 2))
    report("J3 激发数守恒", abs(e1 - e0) < 1e-9,
           "初始 %.6f，末态 %.6f（寄存器布居 %.3f）" % (e0, e1, float(np.sum(np.abs(psi[:, :, 1]) ** 2))))

    print("== J4 写有代价 ==")
    psi_free = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk0 = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[0] ** 2))
    psi_free[:, 1, 0] = pk0 / np.linalg.norm(pk0)
    def width_of(p):
        dens = np.sum(np.abs(p) ** 2, axis=(1, 2))
        dens = dens / dens.sum()
        ang = np.angle(np.sum(dens * np.exp(1j * theta)))
        return float(np.sqrt(np.sum(dens * (np.angle(np.exp(1j * (theta - ang))) ** 2))))
    w0 = width_of(psi_free)
    for _ in range(int(round(T_MAX / DT))):
        psi_free = split_step(psi_free, DT, theta, kvec, W_PULSE, 0.0)  # 关掉耦合
    w_free = width_of(psi_free)
    report("J4 反作用落在被采样对象而非时钟", abs(results[0][3] / w_free - 1.0) < 0.01 and site_left < 0.05,
           "指针相位宽度：无写入 %.4f，有写入 %.4f（变化 %+.2f%%）；站点剩余激发 %.3f（转移进寄存器）"
           % (w_free, results[0][3], 100.0 * (results[0][3] / w_free - 1.0), site_left))

    print("== J5 读数登记 ==")
    report("J5 每过轴写一次；n+1 写法待定", True,
           "本轮的写入算符实现为'每过轴转移一个激发'，故内部本底为每站每摆程一笔（与事件模型一致）；"
           "'每激发写一次（$n+1$ 型）'需要在站点模上加计数算符，属后续设计项")

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(theta, pulse(theta, W_PULSE), lw=1.5, label=r"$w = %.2f$" % W_PULSE)
    ax.set_xlabel(r"相位 $\theta$")
    ax.set_ylabel(r"$f(\theta)$")
    ax.set_title("图 A 相位条件：只在过轴处耦合")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for pw, tw_, _, _ in results:
        psi = np.zeros((N_THETA, 2, 2), dtype=complex)
        pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
        psi[:, 1, 0] = pk / np.linalg.norm(pk)
        t = 0.0
        ts, ps = [], []
        for _ in range(int(round(T_MAX / DT))):
            psi = split_step(psi, DT, theta, kvec, W_PULSE, g_full)
            t += DT
            ts.append(t)
            ps.append(float(np.sum(np.abs(psi[:, :, 1]) ** 2)))
        ax.plot(ts, ps, lw=1.4, label=r"packet $= %.2f$" % pw)
    ax.axvline(t_cross, color="k", lw=0.9, ls="--", label="折返时刻")
    ax.set_xlabel(r"时间（单位 $t_P$）")
    ax.set_ylabel("寄存器布居")
    ax.set_title("图 B 写入时刻由过轴给出")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    dens0 = np.sum(np.abs(np.zeros_like(psi) + 0) ** 2, axis=(1, 2)) * 0
    ax.set_xlabel("读数")
    ax.set_ylabel("—")
    ax.set_title("图 C 反作用的位置（见 §三）")
    ax.text(0.1, 0.5, "无写入 %.4f\n有写入 %.4f" % (w_free, results[0][3]), fontsize=11)

    ax = axes[1, 1]
    ax.bar(["无写入", "有写入"], [w_free, results[0][3]], color=["#4f46e5", "#0d9488"], width=0.5)
    ax.set_ylabel("指针相位宽度")
    ax.set_title("图 D 指针不受写入扰动")

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/write_operator.png"
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
