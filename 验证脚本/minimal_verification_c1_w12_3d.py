# -*- coding: utf-8 -*-
"""
minimal_verification_c1_w12_3d.py

[C1-6 W1/W2 三维版] 保护结构在三维线缺陷构型上的复核
（判据 W1-1 至 W1-4、W2-1 至 W2-3）

背景：攻坚计划（三维复合位错计划）§10.3 的波性侧判据 W1（扇区分解）与
W2（弦形跨扇区算符）已在二维最小模型上通过（判据 V-1 至 V-4）；三维构型上
的同组判据待做。本轮把同组判据搬到 C1-6 三维版（B 轮）的线缺陷构型上复核。

模型（与 B 轮同口径）：横截面 N^2 = 128^2，场 u = u0 + theta/2 pi，
u0 = IX + IY（均匀缠绕背景），theta = atan2(y - y0, x - x0)；
线缺陷 = 该二维涡旋沿 z 向延伸（NZ = 8 层，层间独立）。绕数按 mod 1 的
环绕差之和定义（plaquette 口径）。本判据只做结构读数，不含弛豫与动力学。

判据
W1-1 逐切片边界性：每层 W_area = W_boundary（机器精度）、为整数、恒为 1
W1-2 窗口测量（C1-0 装置口径）：窗口含芯给 1，不含芯给 0
W1-3 局域体算符不改扇区：100 组三维球扰动 |dW| = 0 精确；裕度扫描 d > r 恒 0
W1-4 扇区内重分布：体内插入涡旋对：dW = 0 精确且 plaquette 非零计数上升
W2-1 局域轨迹不改标签：芯沿内部路径运动（距边界 >= 8 格），W 恒为 1
W2-2 弦抵达边界才改标签：右、上、对角、折线四路径越界 1 到 0；反向 0 到 1
W2-3 逐切片作用：只在 z = 4 层越界：W(z=4) = 0，其余层保持 1

运行：python 验证脚本/minimal_verification_c1_w12_3d.py（确定性，约 30 秒）
"""
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
N = 128
NZ = 8
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/c1_w12_3d.png"


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def field2d(x0, y0):
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    theta = np.arctan2(IY - y0, IX - x0)
    return IX + IY + theta / TWO_PI


def field3d(x0, y0):
    return np.repeat(field2d(x0, y0)[:, :, None], NZ, axis=2)


def plaq3d(u):
    dx = wrap1(u[1:, :-1, :] - u[:-1, :-1, :])
    dx2 = wrap1(u[1:, 1:, :] - u[:-1, 1:, :])
    dy = wrap1(u[:-1, 1:, :] - u[:-1, :-1, :])
    dy2 = wrap1(u[1:, 1:, :] - u[1:, :-1, :])
    w = dx + dy2 - dx2 - dy
    err = float(np.abs(w - np.rint(w)).max())
    Wz = np.rint(w).sum(axis=(0, 1)).astype(int)
    return Wz, err, w


def loop_winding(u2):
    loop = np.concatenate([u2[0, ::-1], u2[::-1, 0], u2[-1, :], u2[:, -1]])
    return float(np.sum(wrap1(np.diff(np.append(loop, loop[0])))))


print("=" * 78)
print("C1-6 W1/W2 三维版：保护结构在三维线缺陷构型上的复核")
print("=" * 78)

# 定向约定自查（面积求和与边界回路求和必须在同一约定下）
u0 = field3d(64.0, 64.0)
Wz0, err0, _ = plaq3d(u0)
b_conv = loop_winding(u0[:, :, 0])
print()
print("[0] 定向自查：面积求和 W = %d，边界回路（V 约定）= %+.1f" % (Wz0[0], b_conv))
print("     plaquette 取整误差 = %.1e" % err0)

# W1-1 逐切片边界性
print()
print("[W1-1] 逐切片边界性（面积求和 = 边界回路求和）")
for (x0, y0, tag) in ((64.0, 64.0, "芯在中心"), (24.0, 96.0, "芯在偏心")):
    u = field3d(x0, y0)
    Wz, err, _ = plaq3d(u)
    diffs = [abs(int(Wz[k]) - loop_winding(u[:, :, k])) for k in range(NZ)]
    print("     %s：W_area 全层 = %s；边界回路差最大值 = %.2e；取整误差 = %.1e"
          % (tag, str(Wz.tolist()), max(diffs), err))

# W1-2 窗口测量
print()
print("[W1-2] 窗口测量（C1-0 装置口径）：窗口含芯给 1，不含芯给 0")
u = field3d(64.0, 64.0)
wins = (((48, 80, 48, 80), "含芯窗口 [48,80]^2"),
        ((32, 96, 32, 96), "含芯窗口 [32,96]^2"),
        ((0, 40, 0, 40), "不含芯窗口 [0,40]^2"),
        ((88, 127, 88, 127), "不含芯窗口 [88,127]^2"))
for (x1, x2, y1, y2), tag in wins:
    v = loop_winding(u[x1:x2 + 1, y1:y2 + 1, 0])
    print("     %s：W_win = %+.0f" % (tag, v))
u2 = field3d(24.0, 96.0)
v_in = loop_winding(u2[8:40, 80:112, 0])
v_out = loop_winding(u2[40:72, 40:72, 0])
print("     偏心芯 (24,96)：含芯窗口 [8,40]x[80,112] 给 %+.0f；不含芯窗口 [40,72]^2 给 %+.0f"
      % (v_in, v_out))

# W1-3 局域体算符不变性
print()
print("[W1-3] 局域体算符不改扇区（三维球扰动，不触边界）")
rng = np.random.default_rng(20260920)
i = np.arange(N, dtype=float)
Iz = np.arange(NZ, dtype=float)
X, Y, Z = np.meshgrid(i, i, Iz, indexing="ij")
base = field3d(64.0, 64.0)
W0, _, _ = plaq3d(base)
W0z = int(W0[0])
worst = 0
for k in range(100):
    r = float(rng.integers(4, 11))
    cx = float(rng.integers(int(r) + 4, N - int(r) - 5))
    cy = float(rng.integers(int(r) + 4, N - int(r) - 5))
    cz = float(rng.integers(1, NZ - 1))
    sub = base.copy()
    rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    m = rr <= r
    sub[m] += rng.uniform(-0.5, 0.5) * 0.5
    Wp, _, _ = plaq3d(sub)
    worst = max(worst, abs(int(Wp[0]) - W0z), int(np.abs(Wp - W0).max()))
print("     100 组随机三维球扰动（r = 4 至 10，中心距边界 >= 4）：|dW| 最大 = %d（期望 0）" % worst)
print("     裕度扫描（球心横向距离 d、半径 r）：")
for r in (4, 8, 12):
    for d in (r + 2, r + 8, 3 * r + 4):
        cx = float(N - 1 - d)
        cy = float(N // 2)
        cz = 1.0
        sub = base.copy()
        rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        sub[rr <= r] += 0.3
        Wp, _, _ = plaq3d(sub)
        print("        r = %2d，d = %3d：dW = %+d" % (r, d, int(Wp[0]) - W0z))

# W1-4 扇区内重分布
print()
print("[W1-4] 扇区内重分布：体内插入涡旋对")
sep = 10.0
rad = 28.0
xc, yc = 64.0, 64.0
i2 = np.arange(N, dtype=float)
X2, Y2 = np.meshgrid(i2, i2, indexing="ij")
X2 = X2[:, :, None]; Y2 = Y2[:, :, None]
th1 = np.arctan2(Y2 - (yc + sep / 2.0), X2 - xc)
th2 = np.arctan2(Y2 - (yc - sep / 2.0), X2 - xc)
pair = (th1 - th2) / TWO_PI
rr = np.sqrt((X2 - xc) ** 2 + (Y2 - yc) ** 2)
win = np.where(rr < rad, np.cos(0.5 * math.pi * np.minimum(rr / rad, 1.0)) ** 2, 0.0)
sub = base + win * pair
Wp, errp, wp = plaq3d(sub)
n0 = int((np.rint(plaq3d(base)[2]) != 0).sum(axis=(0, 1))[0])
n1 = int((np.rint(wp) != 0).sum(axis=(0, 1))[0])
vals, cnts = np.unique(np.rint(wp)[:, :, 0], return_counts=True)
print("     插入 +- 对（隔 10 格，窗半径 28）：全层 dW = %s；取整误差 = %.1e；非零 plaquette 计数 %d -> %d"
      % (str((Wp - W0).tolist()), errp, n0, n1))
print("     z = 0 层 plaquette 取值分布：%s" % str(list(zip(vals.tolist(), cnts.tolist()))))

# W2-1 局域轨迹
print()
print("[W2-1] 局域轨迹不改标签（芯在内部路径上运动，距边界 >= 8 格）")
path = ((64, 64), (56, 56), (72, 56), (72, 72), (56, 72), (64, 64), (48, 64), (64, 48))
vals = []
for (x0, y0) in path:
    u = field3d(float(x0), float(y0))
    Wz, _, _ = plaq3d(u)
    vals.append(int(Wz[0]))
print("     路径点 %s" % str([tuple(p) for p in path]))
print("     读数 W = %s（期望恒 1）" % str(vals))

# W2-2 弦抵达边界
print()
print("[W2-2] 弦抵达边界才改标签（四路径越界 1 到 0；反向 0 到 1）")
def scan(tag, pts):
    vals = []
    for (x0, y0) in pts:
        u = field3d(float(x0), float(y0))
        Wz, _, _ = plaq3d(u)
        vals.append(int(Wz[0]))
    print("     %s：W 序列 = %s" % (tag, str(vals)))

scan("右方向粗扫 t = 0..72 步 8", [(64 + t, 64) for t in range(0, 73, 8)])
scan("右方向细扫 x0 = 120..136", [(x0, 64) for x0 in range(120, 137)])
scan("上方向细扫 y0 = 120..136", [(64, y0) for y0 in range(120, 137)])
scan("对角细扫 t = 112..132 步 2", [(t, t) for t in range(112, 133, 2)])
scan("折线（右至 x0=124 再上）", [(64 + t, 64) for t in range(0, 61, 20)] + [(124, 64 + s) for s in range(10, 73, 10)])
scan("反向：x0 = 134 回 118", [(x0, 64) for x0 in range(134, 117, -2)])

# W2-3 逐切片作用
print()
print("[W2-3] 逐切片作用：只在 z = 4 层越界")
lay = base.copy()
lay[:, :, 4] = field2d(134.0, 64.0)
Wz, _, _ = plaq3d(lay)
print("     W(z) = %s（期望 z = 4 为 0，其余为 1）" % str(Wz.tolist()))

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

ax = axes[0, 0]
xs = list(range(120, 137))
ws = []
for x0 in xs:
    u = field3d(float(x0), 64.0)
    Wz, _, _ = plaq3d(u)
    ws.append(int(Wz[0]))
ax.step(xs, ws, where="post")
ax.axvline(127.5, color="r", ls=":", lw=1, label="格点边界 127.5")
ax.set_xlabel("芯位 x0（右方向）")
ax.set_ylabel("W")
ax.set_title("(a) 越界扫描：标签只在弦抵达边界时跳变")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.axhline(0, color="k", lw=1)
ax.plot(np.arange(1, 101), np.zeros(100), "o", ms=3, label="100 组球扰动")
ax.set_xlabel("扰动编号")
ax.set_ylabel("dW")
ax.set_ylim(-1.5, 1.5)
ax.set_title("(b) 局域体算符：|dW| 恒为 0")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
labels = ["[48,80]^2", "[32,96]^2", "[0,40]^2", "[88,127]^2"]
vals = []
for (x1, x2, y1, y2) in ((48, 80, 48, 80), (32, 96, 32, 96), (0, 40, 0, 40), (88, 127, 88, 127)):
    vals.append(loop_winding(base[x1:x2 + 1, y1:y2 + 1, 0]))
ax.bar(labels, vals)
ax.set_ylabel("W_win")
ax.set_title("(c) 窗口测量：含芯给 1，不含芯给 0")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.bar(np.arange(NZ), [int(v) for v in Wz])
ax.set_xlabel("z 层")
ax.set_ylabel("W(z)")
ax.set_title("(d) 逐切片作用：仅 z = 4 层标签归零")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
print("总运行时间 = %.1f 秒" % (time.time() - T0))
