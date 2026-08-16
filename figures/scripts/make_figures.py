from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.8,
})

WAVE = "#176b87"
PART = "#c7492d"
AXIS = "#b58a28"
INK = "#1f2933"
MUTED = "#66717d"
GRID = "#d9e0e6"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    ax.set_aspect("equal")
    t = np.linspace(0, 2*np.pi, 500)
    ax.plot(np.cos(t), np.sin(t), color=WAVE, lw=2.0, label=r"$S^1$")
    theta = 0.72
    p = np.array([np.cos(theta), np.sin(theta)])
    ax.arrow(0, 0, p[0]*0.82, p[1]*0.82, width=0.012, head_width=0.09,
             head_length=0.12, length_includes_head=True, color=PART, zorder=4)
    ax.scatter(*p, s=42, color=PART, zorder=5)
    ax.scatter(1, 0, s=45, facecolor="white", edgecolor=AXIS, lw=1.6, zorder=5)
    ax.annotate(r"$\Psi(t)=e^{i\theta(t)}$", xy=p, xytext=(0.55, 1.18),
                arrowprops=dict(arrowstyle="->", color=PART, lw=0.9), color=PART, fontsize=11)
    ax.annotate("相位标记点", xy=(1, 0), xytext=(1.18, -0.3),
                arrowprops=dict(arrowstyle="->", color=AXIS, lw=0.9), color=AXIS, fontsize=10)
    ax.text(0.04, 0.04, "$O$", fontsize=12, color=INK)
    ax.text(-1.35, 1.38, "Figure 1  复平面单位圆与连续相位旋转", fontsize=12, weight="bold", color=INK)
    ax.set_xlabel(r"实轴 $\operatorname{Re} z$")
    ax.set_ylabel(r"虚轴 $\operatorname{Im} z$")
    ax.set_xlim(-1.45, 1.55); ax.set_ylim(-1.3, 1.45)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig01_unit_circle")


def fig2():
    fig = plt.figure(figsize=(7.4, 3.8))
    for i, (title, base, marker, color) in enumerate([
        ("(a) 圆心法线：旋转轴", (0, 0, 0), (0, 0, 1), AXIS),
        ("(b) 圆周点法线：平行但非旋转轴", (1, 0, 0), (1, 0, 1), PART),
    ]):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        u = np.linspace(0, 2*np.pi, 160)
        ax.plot(np.cos(u), np.sin(u), 0, color=WAVE, lw=1.7)
        z = np.linspace(-1.1, 1.1, 30)
        ax.plot(np.full_like(z, base[0]), np.full_like(z, base[1]), z, color=color, lw=2.2)
        ax.scatter([base[0]], [base[1]], [0], color=color, s=42)
        ax.quiver(*base, 0, 0, 0.65, color=color, arrow_length_ratio=0.13, lw=1.5)
        if i == 0:
            ax.scatter([0], [0], [0], color=INK, s=30)
            ax.text(0, 0, 0.13, "$O$", color=INK)
        else:
            ax.text(1.05, 0, 0.12, "$P$", color=INK)
        ax.set_title(title, fontsize=10, pad=8, color=INK)
        ax.set_xlim(-1.35, 1.45); ax.set_ylim(-1.25, 1.25); ax.set_zlim(-1.15, 1.15)
        ax.set_box_aspect((1, 1, 1.0)); ax.set_axis_off()
        ax.view_init(elev=18, azim=-58)
    fig.suptitle("Figure 2  圆心法线与圆周点法线的几何区分", fontsize=12, weight="bold", color=INK, y=0.98)
    fig.text(0.5, 0.03, "法线负责空间路径；圆周标记点负责满周闭合的时间触发", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "fig02_normal_axis")


def fig3():
    fig = plt.figure(figsize=(7.6, 4.1))
    th = np.linspace(0, 8*np.pi, 700)
    z = th/(2*np.pi)
    for i, solid in enumerate([False, True]):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        if solid:
            rr = np.linspace(0, 1, 16)
            T, R = np.meshgrid(th, rr)
            X = R*np.cos(T); Y = R*np.sin(T); Z = T/(2*np.pi)
            ax.plot_surface(X, Y, Z, color=WAVE, alpha=0.16, linewidth=0, shade=True)
            ax.plot_wireframe(X[::3], Y[::3], Z[::3], color=WAVE, alpha=0.35, rstride=1, cstride=12, lw=0.35)
            label = r"$D^2\times\mathbb{R}$"
            dim = r"$\dim_H=3$"
        else:
            ax.plot(np.cos(th), np.sin(th), z, color=WAVE, lw=1.8)
            label = r"$S^1\times\mathbb{R}$"
            dim = r"$\dim_H=2$"
        ax.plot([0, 0], [0, 0], [0, 4], color=AXIS, lw=1.2, alpha=0.8)
        ax.set_title(f"({'b' if solid else 'a'}) {label}  {dim}", fontsize=11, pad=8)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(0, 4)
        ax.set_box_aspect((1, 1, 1.5)); ax.set_axis_off(); ax.view_init(elev=22, azim=-58)
    fig.suptitle("Figure 3  波性边界轨迹与完整物侧状态空间", fontsize=12, weight="bold", color=INK)
    fig.text(0.5, 0.035, "左：边界螺旋面；右：包含径向承载的实心状态空间", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    save(fig, "fig03_state_space")


def fig4():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
    tau = np.linspace(0, 5, 600)
    ax1.plot(tau, 0.8*tau, color=WAVE, lw=2, label=r"连续 $Z_{\rm wave}(\tau)=c\tau$")
    ax1.scatter(np.arange(6), 0.8*np.arange(6), color=PART, s=25, zorder=3, label="满周闭合采样点")
    ax1.set_ylabel(r"$Z$")
    ax1.set_title("连续本体层与离散现象层的采样对偶", fontsize=11, weight="bold")
    ax1.legend(frameon=False, loc="upper left", fontsize=9)
    ax1.grid(True, color=GRID, lw=0.6)
    n = np.arange(0, 6)
    T = n
    Z = 0.8*n
    ax2.step(T, Z, where="post", color=PART, lw=2, label=r"离散 $Z_{\rm particle}(T_n)=l_P n$")
    ax2.scatter(T, Z, color=PART, s=25, zorder=3)
    for k in range(1, 6):
        ax2.annotate(r"$l_P$", xy=(k-0.5, 0.8*k), xytext=(k-0.5, 0.8*k+0.18), ha="center", color=PART, fontsize=9)
    ax2.set_xlabel(r"时间参数：$\tau$（数学时间）或 $T_n=n t_P$（物理时间）")
    ax2.set_ylabel(r"$Z$")
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    ax2.grid(True, color=GRID, lw=0.6)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 5); ax.set_ylim(-0.1, 4.5)
    fig.tight_layout()
    save(fig, "fig04_sampling_duality")


def fig5():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    steps = 12
    x0, y0 = 0.0, 0.0
    for mass, color, label, bend in [(0.0, WAVE, r"$m=0$：法线保持平行", 0.0), (0.16, PART, r"$m>0$：每步偏转 $\delta\theta=m/m_P$", 0.055)]:
        x, y = x0, y0
        angle = np.pi/2
        pts = [(x, y)]
        for k in range(steps):
            angle += bend
            x += np.cos(angle); y += np.sin(angle)
            pts.append((x, y))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, marker="o", ms=3.5, label=label)
        for k in range(0, steps+1, 3):
            ax.annotate(f"{k}", pts[k], xytext=(4, 4), textcoords="offset points", color=color, fontsize=8)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.annotate("法线方向累积偏转", xy=(0.3, 7.1), xytext=(2.0, 7.7), color=PART,
                arrowprops=dict(arrowstyle="->", color=PART, lw=1.0), fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("跃迁步数 $n$")
    ax.set_ylabel("空间位置投影")
    ax.set_title("Figure 5  质量参数引起的离散法线偏转（模型示意）", fontsize=12, weight="bold", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.text(0.02, 0.02, r"$\delta\theta=m/m_P$；图中曲线仅表示模型内的累积几何关系", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xlim(-1.2, 8.0); ax.set_ylim(-0.5, 8.5); ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    save(fig, "fig05_mass_deflection")


def fig6():
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [(0.3, 1.05, "复平面\n相位旋转", WAVE), (2.1, 1.05, "满周闭合\n事件", AXIS), (3.9, 1.05, "圆心法线\n离散跃迁", PART), (5.7, 1.05, "空间像素\n累积", PART), (7.5, 1.05, "宏观时空\n连续与弯曲", WAVE)]
    for x, y, text, color in boxes:
        rect = Rectangle((x, y), 1.45, 0.9, facecolor="white", edgecolor=color, lw=1.7)
        ax.add_patch(rect); ax.text(x+0.725, y+0.45, text, ha="center", va="center", fontsize=10, color=INK)
    for x in [1.75, 3.55, 5.35, 7.15]:
        ax.add_patch(FancyArrowPatch((x, 1.5), (x+0.32, 1.5), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=INK))
    ax.text(1.0, 2.35, "波性侧：连续、可逆、数学时间 $\\tau$", ha="center", color=WAVE, fontsize=9)
    ax.text(6.6, 0.42, "粒性侧：离散、不可逆、物理时间 $T_n=n t_P$", ha="center", color=PART, fontsize=9)
    ax.annotate("模型中的几何对应关系", xy=(5.0, 2.0), xytext=(5.0, 2.7), ha="center", color=MUTED, fontsize=9,
                arrowprops=dict(arrowstyle="-[", color=MUTED, lw=0.8))
    ax.set_title("Figure 6  元相子螺旋阶梯几何的理论结构总览", fontsize=12, weight="bold", color=INK, pad=12)
    fig.tight_layout()
    save(fig, "fig06_theory_overview")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"generated 18 files in {OUT}")
