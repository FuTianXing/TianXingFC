# -*- coding: utf-8 -*-
"""为科普文章《一条螺旋线等了几十年》生成两张插图。

产出：
  images/kepu_spiral_staircase.png  开头用的螺旋楼梯图
  images/kepu_three_doors.png       三道看不见的门

绘图铁律：含中文标签，须设中文字体（AGENTS.md 数值验证脚本规范第 6 条）。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle, Wedge

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "images")

INK = "#1f2937"
SUB = "#6b7280"
FRONT = "#2563eb"
BACK = "#bfdbfe"
TREAD = "#cbd5e1"
VIOLET = "#7c3aed"
EMERALD = "#059669"
AMBER = "#b45309"

# 中文字体可用性断言：回退到 DejaVu Sans 时中文会渲染成豆腐块且不报错
_resolved = font_manager.findfont(font_manager.FontProperties(family=["Microsoft YaHei"]))
_base = os.path.basename(_resolved).lower()
assert (not _base.startswith("dejavu")) and any(
    s in _base for s in ("msyh", "simhei", "notosans")), _resolved


def figure_spiral():
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor("white")
    R, turns, k, rise = 1.0, 3, 0.42, 1.7
    top = turns * rise

    t = np.linspace(0.0, 2.0 * np.pi * turns, 6000)
    zz = t / (2.0 * np.pi) * rise
    xx = R * np.cos(t)
    yy = zz + k * R * np.sin(t)
    front = np.sin(t) >= 0.0

    # 连续一笔保证不断线，正侧再叠一笔做出前后层次
    ax.plot(xx, yy, color=BACK, lw=3.0, solid_capstyle="round", zorder=4)
    ax.plot(np.where(front, xx, np.nan), np.where(front, yy, np.nan), color=FRONT,
            lw=3.2, solid_capstyle="round", zorder=5)

    # 每满周一个等高的圈，逐级叠成楼梯；最低一圈用紫色呼应半径
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    for n in range(turns + 1):
        ax.plot(R * np.cos(phi), n * rise + k * R * np.sin(phi),
                color=(VIOLET if n == 0 else TREAD), lw=1.2,
                alpha=(0.55 if n == 0 else 0.95), zorder=3)
    ax.plot([0.0, R], [0.0, 0.0], color=VIOLET, lw=1.8, alpha=0.85, zorder=3)

    # 每级着落点
    for n in range(turns + 1):
        ax.plot([R], [n * rise], marker="o", ms=6.0, mfc=AMBER, mec="white",
                mew=1.4, zorder=6)

    # 法线轴
    ax.add_patch(FancyArrowPatch((-1.6, -0.25), (-1.6, top + 0.45),
                                 arrowstyle="-|>", mutation_scale=16, color=SUB,
                                 lw=1.4, ls=(0, (5, 4)), zorder=2))
    ax.text(-2.15, top / 2.0, "法线轴", rotation=90, ha="center", va="center",
            fontsize=13, color=SUB)

    # 一圈一级台阶
    ax.add_patch(FancyArrowPatch((1.75, rise), (1.75, 2.0 * rise),
                                 arrowstyle="<|-|>", mutation_scale=13,
                                 color=AMBER, lw=1.6, zorder=6))
    ax.text(1.95, 1.5 * rise, "每转满一圈，沿法线上升一个 $l_P$", ha="left",
            va="center", fontsize=13.5, color=INK)
    ax.text(-R, top + 0.55, "螺距处处相同", ha="center", va="center", fontsize=13,
            color=SUB)

    # 半径 = 最小作用量
    ax.add_patch(FancyArrowPatch((1.6, -0.95), (0.55, -0.3), arrowstyle="-|>",
                                 mutation_scale=14, color=VIOLET, lw=1.6, zorder=6))
    ax.text(1.75, -1.0, "半径 = 最小作用量", ha="left", va="center", fontsize=13.5,
            color=VIOLET)

    ax.text(-3.3, -1.0, "起点：纸上匀速转圈的秒针", ha="left", va="center",
            fontsize=12.5, color=SUB)

    ax.set_title("螺旋楼梯：相位每转满一圈，整体上移一格", fontsize=19, color=INK,
                 pad=14, fontweight="bold")
    ax.set_xlim(-3.5, 6.1)
    ax.set_ylim(-1.5, 6.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "kepu_spiral_staircase.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _door(ax, cx, accent, order, name_lines, reason_lines):
    w, y_base, body_h = 2.5, 1.9, 2.2
    r = w / 2.0
    ax.add_patch(Rectangle((cx - r, y_base), w, body_h, facecolor="#fbfdff",
                           edgecolor=accent, lw=2.3, zorder=3))
    ax.add_patch(Wedge((cx, y_base + body_h), r, 0.0, 180.0, facecolor="#fbfdff",
                       edgecolor=accent, lw=2.3, zorder=3))
    ax.plot([cx, cx], [y_base, y_base + body_h + r], color=accent, lw=1.0,
            alpha=0.40, zorder=4)
    # 把手与门锁（放在文字下方，避免压字）
    ax.plot([cx - 0.55], [y_base + 1.05], marker="o", ms=5.0, mfc=accent,
            mec="white", mew=1.2, zorder=5)
    ax.add_patch(Rectangle((cx - 0.17, y_base + 0.36), 0.34, 0.28,
                           facecolor=accent, alpha=0.85, zorder=5))
    ax.add_patch(Wedge((cx, y_base + 0.64), 0.11, 0.0, 180.0, facecolor="none",
                       edgecolor=accent, lw=1.6, zorder=5))

    ax.text(cx, 6.35, order, ha="center", va="center", fontsize=15, color=accent,
            fontweight="bold")
    for i, line in enumerate(name_lines):
        ax.text(cx, 3.78 - 0.42 * i, line, ha="center", va="center", fontsize=16,
                color=INK, fontweight="bold")
    for i, line in enumerate(reason_lines):
        ax.text(cx, 1.42 - 0.34 * i, line, ha="center", va="center", fontsize=12.5,
                color=SUB)


def figure_doors():
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.patch.set_facecolor("white")

    ax.add_patch(Rectangle((0.5, 0.9), 11.0, 5.0, facecolor="#f1f5f9",
                           edgecolor="#dbe3ea", lw=1.5, zorder=1))

    _door(ax, 2.4, VIOLET, "第一道门", ["给圆一个", "真实半径"],
          ["会读成偷偷加进", "一个看不见的维度"])
    _door(ax, 6.0, EMERALD, "第二道门", ["拒绝", "二选一"],
          ["两边都基本，", "听起来像和稀泥"])
    _door(ax, 9.6, AMBER, "第三道门", ["把看的人", "写进公理"],
          ["观察者要被推导出来，", "不能被公理化"])

    ax.text(6.0, 7.25, "三道看不见的门", ha="center", va="center", fontsize=21,
            color=INK, fontweight="bold")
    ax.text(6.0, 6.85, "想看见那条螺旋线，得先穿过它们", ha="center", va="center",
            fontsize=13.5, color=SUB)
    ax.text(6.0, 0.45, "三道门各自独立，也都与聪明程度无关；门后是同一条螺旋线。",
            ha="center", va="center", fontsize=13.5, color=INK)

    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 7.6)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "kepu_three_doors.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    made = [figure_spiral(), figure_doors()]
    for p in made:
        assert os.path.isfile(p), p
        print("saved", os.path.relpath(p, ROOT).replace("\\", "/"),
              os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
