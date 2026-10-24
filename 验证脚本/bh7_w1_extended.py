# -*- coding: utf-8 -*-
"""BH7-W1 开域：更大 h0、更长时程、更大格点的三轴外推。

判据（先立判据，再实现；承接 注记-BH7-W1 的边界声明）：
  W1b-1 对照一致：N3=12 臂复现 W1 读数（Psi(h0=1)=0.994434、Psi(h0=4)=0.983629，容差 1e-3）。
  W1b-2 格点（报告）：N3=16、20 对 12 的 Psi 位移（有限尺寸效应）。
  W1b-3 时程（报告）：h0=4 的终态 Psi、R 在 600/1200/2400 步（延迟塌缩检查）。
  W1b-4 幅度（报告）：h0 至 32 的 Psi、R、键级缺陷密度 rho_b（阈值搜索：Psi < 0.9 或 R < 0.9）。

模型：BH3-3 摆程层协议（同 W1；滴半径 2、写位为滴表面、登记符号臂、T_beat=12、dt=0.02）。
确定性。图存 D:/software/TianXingFC/images/bh7_w1_extended.png。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== BH7-W1 开域：三轴外推 ====")
T_beat = 12.0; DT = 0.02; Rd = 2

def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def run(N3, h0, lam, steps, seed=1, record=None):
    c3 = N3 // 2
    ii, jj, kk = np.indices((N3, N3, N3))
    base = np.pi * (ii + jj + kk)
    stagger = (ii + jj + kk) % 2
    droplet = ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) <= Rd ** 2
    sig = np.where(droplet, 1.0, 1.0 - 2.0 * stagger)
    ws = droplet & (((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) >= (Rd - 1) ** 2)
    rng = np.random.RandomState(seed)
    th = base + 1e-6 * rng.standard_normal(base.shape)
    h = np.zeros_like(th); h[ws] = h0
    drive = sig * (2.0 * np.pi / T_beat) * DT
    hist = []
    for t in range(steps):
        g = (np.sin(th - np.roll(th, 1, axis=0)) + np.sin(th - np.roll(th, -1, axis=0)) +
             np.sin(th - np.roll(th, 1, axis=1)) + np.sin(th - np.roll(th, -1, axis=1)) +
             np.sin(th - np.roll(th, 1, axis=2)) + np.sin(th - np.roll(th, -1, axis=2)))
        th = th - DT * (-g + lam * np.sin(th) * h) + drive
        if record is not None and (t + 1) % record == 0:
            hist.append((t + 1, readouts(th, base)))
    return th, base, hist

def readouts(th, base):
    d = wrap(th - base)
    psi = float(np.abs(np.mean(np.exp(1j * d))))
    R = float(np.mean(np.abs(d) < np.pi / 2.0))
    nbd = np.zeros(th.shape, dtype=int); ntot = 0
    for ax in range(3):
        for s_ in (1, -1):
            bad = np.abs(wrap(wrap(th - np.roll(th, s_, axis=ax)) - np.pi)) > np.pi / 2.0
            nbd += bad.astype(int); ntot += bad.size
    return dict(psi=psi, R=R, rho_b=float(nbd.sum()) / ntot)

# W1b-1 对照
t12a, b12, _ = run(12, 1.0, 1.0, 600)
t12b, _, _ = run(12, 4.0, 1.0, 600)
r1a = readouts(t12a, b12); r1b = readouts(t12b, b12)
print("  [W1b-1] N3=12：Psi(h0=1) = %.6f（W1 0.994434）；Psi(h0=4) = %.6f（W1 0.983629）" % (r1a["psi"], r1b["psi"]))
check("W1b-1", abs(r1a["psi"] - 0.994434) <= 1e-3 and abs(r1b["psi"] - 0.983629) <= 1e-3, "复现 W1 读数")

# W1b-2 格点
print("  [W1b-2] 格点效应：")
grid_rows = []
for N3 in (12, 16, 20):
    for h0 in (1.0, 4.0):
        th, bs, _ = run(N3, h0, 1.0, 600)
        rr = readouts(th, bs)
        grid_rows.append((N3, h0, rr))
        print("       N3=%2d h0=%.0f：Psi = %.6f  R = %.5f" % (N3, h0, rr["psi"], rr["R"]))
check("W1b-2", True, "格点读数已报告")

# W1b-3 时程
print("  [W1b-3] 时程（N3=12、h0=4）：")
th, bs, hist = run(12, 4.0, 1.0, 2400, record=12)
for st, rr in hist:
    if st in (600, 1200, 2400):
        print("       step=%4d：Psi = %.6f  R = %.5f" % (st, rr["psi"], rr["R"]))
check("W1b-3", True, "时程读数已报告（延迟塌缩检查）")

# W1b-4 幅度
print("  [W1b-4] 幅度（N3=12、600 步）：")
amp_rows = []
for h0 in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    th, bs, _ = run(12, h0, 1.0, 600)
    rr = readouts(th, bs)
    amp_rows.append((h0, rr))
    print("       h0=%5.1f：Psi = %.6f  R = %.5f  rho_b = %.5f" % (h0, rr["psi"], rr["R"], rr["rho_b"]))
low = [x for x in amp_rows if x[1]["psi"] < 0.9 or x[1]["R"] < 0.9]
check("W1b-4", True, "幅度读数已报告（低于 0.9 的档数 = %d）" % len(low))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
for N3, mk in ((12, "o-"), (16, "s-"), (20, "^-")):
    ys = [rr["psi"] for (n, h0, rr) in grid_rows if n == N3]
    ax.plot([1.0, 4.0], ys, mk, label="N3=%d" % N3)
ax.set_xlabel("h0"); ax.set_ylabel("Psi"); ax.set_title("(a) 格点效应"); ax.legend(fontsize=8)
ax = axs[1]
ts = [st for st, _ in hist]; ax.plot(ts, [rr["psi"] for _, rr in hist], "-", label="Psi")
ax.plot(ts, [rr["R"] for _, rr in hist], "--", label="R")
ax.set_xlabel("步"); ax.set_ylabel("值"); ax.set_title("(b) 时程（h0=4，2400 步）"); ax.legend(fontsize=8)
ax = axs[2]
ax.semilogx([h0 for h0, _ in amp_rows], [rr["psi"] for _, rr in amp_rows], "o-", label="Psi")
ax.semilogx([h0 for h0, _ in amp_rows], [rr["R"] for _, rr in amp_rows], "s--", label="R")
ax.axhline(0.9, color="tab:red", ls=":", lw=1)
ax.set_xlabel("h0"); ax.set_ylabel("值"); ax.set_title("(c) 幅度外推（h0 至 32）"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_w1_extended.png", dpi=140)
print("  图已存 images/bh7_w1_extended.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
