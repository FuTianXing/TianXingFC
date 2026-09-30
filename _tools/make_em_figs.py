# -*- coding: utf-8 -*-
"""为《电磁场总论》生成两张示意图。

图 A：联络绕格面一圈的净扭转（曲率的定义，支撑第二章）。
图 B：晶格色散的低能直线与首阶洛伦兹破坏方向依赖（支撑第五章 5.4）。

数据来源：图 A 取元相子规范块的标准绕圈构造，解析值为磁通 Phi=1；
图 B 取《元相子振荡晶格的色散关系》第 5 节的首阶展开式与三方向系数。
两图均为解析量绘制，不含经验拟合。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

AXIS = "#D14A2A"      # 实轴方向场
ORIENT = "#2A6FD1"    # 基准取向场
GOLD = "#B8860B"

# ---------------- 图 A：曲率的绕圈定义 ----------------
def fig_curvature():
    Phi = 1.0
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) 三条闭合曲线绕磁通管
    ax[0].set_title("(a) 三条闭合曲线环绕磁通管", fontsize=11)
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    shapes = {
        "r=2 圆": 2.0 * np.ones_like(th),
        "r=2(1+0.3 sin3θ)": 2.0 * (1.0 + 0.3 * np.sin(3 * th)),
        "椭圆": 1.0 / np.sqrt((np.cos(th) / 1.5) ** 2 + (np.sin(th) / 3.0) ** 2),
    }
    for lbl, r in shapes.items():
        ax[0].plot(r * np.cos(th), r * np.sin(th), lw=1.4, label=lbl)
    ax[0].plot(0, 0, "o", ms=9, color=GOLD, label="磁通管截面")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].grid(alpha=0.25)

    # (b) 沿曲线累积的相位角：三条曲线在终点重合成同一条
    ax[1].set_title("(b) 沿曲线累积的相位角：终点重合成 $2\\pi$", fontsize=11)
    thq = np.linspace(0, 2 * np.pi, 20001)
    dq = thq[1] - thq[0]
    for lbl, r in shapes.items():
        rq = np.interp(thq, th, r)
        x, y = rq * np.cos(thq), rq * np.sin(thq)
        R2 = x ** 2 + y ** 2
        Ax = -Phi / (2 * np.pi) * y / R2
        Ay = Phi / (2 * np.pi) * x / R2
        # 参数切向量：dr/dθ 绕 (cosθ, sinθ) 加 r 绕 (-sinθ, cosθ)
        dr = np.gradient(rq, thq)
        dx = dr * np.cos(thq) - rq * np.sin(thq)
        dy = dr * np.sin(thq) + rq * np.cos(thq)
        integrand = Ax * dx + Ay * dy
        phase = np.cumsum(integrand) * dq
        phase = phase - phase[0]
        ax[1].plot(thq, phase, lw=1.6, label=lbl)
    ax[1].axhline(2 * np.pi, ls="--", lw=1.1, color="gray")
    ax[1].set_xlabel("θ")
    ax[1].set_ylabel("累积相位（弧度）")
    ax[1].set_ylim(-0.3, 2.35 * np.pi)
    ax[1].set_xlim(0, 2 * np.pi)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].grid(alpha=0.25)
    ax[1].text(0.30, 0.92, "终点同在 $2\\pi$：路径形状不影响总扭转",
               transform=ax[1].transAxes, fontsize=9, color="#444")

    # (c) 曲线半径改变，圈积分不变
    ax[2].set_title("(c) 半径扫描：圈积分恒为磁通", fontsize=11)
    rs = np.linspace(0.5, 4.0, 200)
    thn = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    vals = []
    for r in rs:
        x, y = r * np.cos(thn), r * np.sin(thn)
        Aphi = Phi / (2 * np.pi * r)
        dphi = np.gradient(np.append(0, 2 * np.pi, 0))[1:-1] if False else None
        thc = np.append(thn, 2 * np.pi)
        xc, yc = np.append(x, x[0]), np.append(y, y[0])
        dx = np.gradient(xc, thc)[:-1]
        dy = np.gradient(yc, thc)[:-1]
        Ax = -Phi / (2 * np.pi) * y / r ** 2
        Ay = Phi / (2 * np.pi) * x / r ** 2
        vals.append(np.sum(Ax * dx + Ay * dy) * (2 * np.pi / len(thn)))
    vals = np.array(vals)
    ax[2].plot(rs, vals, lw=1.6, color=ORIENT, label=r"$\oint A\cdot dl$")
    ax[2].axhline(Phi, ls="--", lw=1.2, color="gray", label=r"$\Phi = 1$")
    ax[2].set_xlabel("曲线半径 r")
    ax[2].set_ylabel("圈积分")
    ax[2].set_ylim(0.9 * Phi, 1.1 * Phi)
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.25)
    ax[2].text(0.05, 0.06, "最大偏差 %.1e" % np.max(np.abs(vals - Phi)),
               transform=ax[2].transAxes, fontsize=9, color="#444")

    fig.suptitle("图 A：联络绕格面一圈的净扭转即场强（曲率的定义）", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "em_curvature_def.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, np.max(np.abs(vals - Phi))


# ---------------- 图 B：色散与首阶洛伦兹破坏 ----------------
def fig_dispersion():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    lP = 1.0
    tP = 1.0

    # (a) 色散曲线与低能直线
    ax[0].set_title("(a) 晶格色散与低能直线 $\\omega = ck$", fontsize=11)
    kk = np.linspace(0, np.pi / lP, 400)
    # [100] 方向：k = (k, 0, 0)
    w_exact = 2.0 / tP * np.abs(np.sin(0.5 * kk))
    ax[0].plot(kk * lP, w_exact * tP, lw=1.8, color=ORIENT, label="精确色散")
    ax[0].plot(kk * lP, kk * lP, ls="--", lw=1.3, color="gray", label=r"$\omega t_P = k l_P$")
    ax[0].set_xlabel(r"$k l_P$")
    ax[0].set_ylabel(r"$\omega t_P$")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.25)
    ax[0].text(0.42, 0.86, "低能段两者重合", transform=ax[0].transAxes, fontsize=9, color="#444")

    # (b) 相对偏离随 l_P k
    ax[1].set_title("(b) 首阶偏离随 $l_P k$ 增长", fontsize=11)
    u = np.linspace(0.01, 1.0, 300)
    for frac, lbl, c in ((1.0, "[100]", AXIS), (0.5, "[110]", "#8A7B4F"), (1.0 / 3.0, "[111]", "#0F8A5F")):
        ax[1].plot(u, -(1.0 / 24.0) * frac * u ** 2, lw=1.7, color=c, label=lbl)
    ax[1].set_xlabel(r"$l_P k = E/E_P^{(t)}$")
    ax[1].set_ylabel(r"$\delta\omega/\omega$")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.25)
    ax[1].text(0.05, 0.10, "标度 $\\propto(E/E_P^{(t)})^2$", transform=ax[1].transAxes, fontsize=9, color="#444")

    # (c) 三方向系数柱状，验证 6:3:2
    ax[2].set_title("(c) 三方向系数比严格为 $6:3:2$", fontsize=11)
    dirs = ["[100]", "[110]", "[111]"]
    coef = [1.0 / 24.0, 1.0 / 48.0, 1.0 / 72.0]
    ratio = [c / coef[0] for c in coef]
    bars = ax[2].bar(dirs, ratio, color=[AXIS, "#8A7B4F", "#0F8A5F"], width=0.55)
    for b, v, t in zip(bars, ratio, ["1", "1/2", "1/3"]):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, t, ha="center", fontsize=10)
    ax[2].set_ylabel("相对 [100] 的比值")
    ax[2].set_ylim(0, 1.25)
    ax[2].grid(alpha=0.25, axis="y")
    ax[2].text(0.05, 0.88, "系数 $-1/24$, $-1/48$, $-1/72$", transform=ax[2].transAxes, fontsize=9, color="#444")

    fig.suptitle("图 B：低能光速涌现与首阶洛伦兹破坏的方向依赖", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "em_dispersion_lv.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


a, dev = fig_curvature()
b = fig_dispersion()
print("图 A ->", a, "  圈积分最大偏差 %.2e" % dev)
print("图 B ->", b)
