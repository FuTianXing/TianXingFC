# -*- coding: utf-8 -*-
"""BH7-W1：交错序参量读数（结构序破坏的微观判据）。

判据（先立判据，再实现；草案见 内部草稿/判据草案-第二主问题：普朗克质量与晶格损坏的判别（W系列）.md）：
  W1-1 口径入册：三个读数在摆动解卷后定义——相位偏差 d = wrap(theta - base)（终点，摆动相位
       为 2 pi 整数倍时）；交错序参量 Psi = |mean exp(i d)|；接力连续性 R = mean(|d| < pi/2)；
       偏尾 P90 = |d| 的 90 分位。理想基态 Psi = 1、R = 1。
  W1-2 对照锚：无写入（h 恒零）时 Psi >= 0.99，且 lambda = 0 与 1 不可分辨（差 <= 1e-3）；三种子散布 <= 1e-3。
  W1-3 序破坏扫描（报告）：写入幅度 h0 扫描（0.5 至 4）的 Psi、R、P90；判定平滑衰减或阈值型。
  W1-4 登记符号臂对照（报告）：lambda = 1（回拉，裁决二十六）对 lambda = 0 的 Psi 差。
  W1-5 轨迹（报告）：h0 = 1、lambda = 1 的 Psi(t) 曲线（每 12 步采样）。

模型：BH3-3 摆程层协议原样复用（12^3、交错基态 base = pi(i+j+k)、过阻尼弛豫、节拍驱动、
登记符号实现：theta <- theta - dt(-g + lambda sin(theta) h) + drive；T_beat = 12、600 步 = 50 拍、
dt = 0.02、噪声种子固定）。确定性。图存 D:/software/TianXingFC/images/bh7_w1_staggered.png。
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

print("==== BH7-W1 交错序参量读数 ====")
N3 = 12
c3 = N3 // 2
Rd = 2
ii, jj, kk = np.indices((N3, N3, N3))
base = np.pi * (ii + jj + kk)
stagger = (ii + jj + kk) % 2
droplet = ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) <= Rd ** 2
sig_vac0 = 1.0 - 2.0 * stagger
sig_lock0 = np.where(droplet, 1.0, sig_vac0)
write_sites = droplet & (((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) >= (Rd - 1) ** 2)
T_beat = 12.0
DT = 0.02
STEPS = 600

def relax(lam, h0, seed, record=False):
    rng = np.random.RandomState(seed)
    th = base + 1e-6 * rng.standard_normal(base.shape)
    h = np.zeros_like(th)
    if h0 is not None:
        h[write_sites] = h0
    drive = sig_lock0 * (2.0 * np.pi / T_beat) * DT
    traj = []
    for t in range(STEPS):
        g = (np.sin(th - np.roll(th, 1, axis=0)) + np.sin(th - np.roll(th, -1, axis=0)) +
             np.sin(th - np.roll(th, 1, axis=1)) + np.sin(th - np.roll(th, -1, axis=1)) +
             np.sin(th - np.roll(th, 1, axis=2)) + np.sin(th - np.roll(th, -1, axis=2)))
        th = th - DT * (-g + lam * np.sin(th) * h) + drive
        if record and (t + 1) % 12 == 0:
            traj.append((t + 1, unwrapped_readouts(th)))
    return th, traj

def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def unwrapped_readouts(th):
    d = wrap(th - base)
    psi = float(np.abs(np.mean(np.exp(1j * d))))
    R = float(np.mean(np.abs(d) < (np.pi / 2.0)))
    p90 = float(np.percentile(np.abs(d), 90.0))
    return dict(psi=psi, R=R, p90=p90)

# W1-2 对照锚（无写入）
a0 = unwrapped_readouts(relax(0.0, None, 1)[0])
a1 = unwrapped_readouts(relax(1.0, None, 1)[0])
sc = [unwrapped_readouts(relax(1.0, None, s)[0])["psi"] for s in (1, 2, 3)]
print("  [W1-2] 无写入：Psi(lambda=0) = %.6f；Psi(lambda=1) = %.6f；差 = %.2e；种子散布 = %.2e"
      % (a0["psi"], a1["psi"], abs(a0["psi"] - a1["psi"]), max(sc) - min(sc)))
check("W1-2", a0["psi"] >= 0.99 and a1["psi"] >= 0.99 and abs(a0["psi"] - a1["psi"]) <= 1e-3
      and (max(sc) - min(sc)) <= 1e-3, "对照锚：无写入 Psi 饱和且耦合不可分辨")

# W1-3/W1-4 写入扫描
H0 = [0.5, 1.0, 2.0, 4.0]
rows = []
for h0 in H0:
    for lam in (0.0, 1.0):
        rr = unwrapped_readouts(relax(lam, h0, 1)[0])
        rows.append((h0, lam, rr))
        print("  [读数] h0=%.1f lambda=%.0f：Psi=%.6f  R=%.4f  P90=%.4f" % (h0, lam, rr["psi"], rr["R"], rr["p90"]))
psi_o = [rr["psi"] for _, lam, rr in rows if lam == 0.0]
psi_1 = [rr["psi"] for _, lam, rr in rows if lam == 1.0]
mono = all(psi_o[i + 1] <= psi_o[i] + 1e-9 for i in range(len(psi_o) - 1))
check("W1-3", True, "序参量扫描已报告（单调下降 = %s；最小 Psi = %.4f）" % (mono, min(psi_o + psi_1)))
gap = max(abs(a - b) for a, b in zip(psi_o, psi_1))
check("W1-4", True, "登记符号臂对照已报告（lambda=1 对 0 的最大 Psi 差 = %.4f）" % gap)

# W1-5 轨迹
_, traj = relax(1.0, 1.0, 1, record=True)
print("  [W1-5] 轨迹（h0=1、lambda=1）：Psi(t) 每 120 步： " +
      " ".join("%.4f" % x[1]["psi"] for x in traj if x[0] % 120 == 0))
check("W1-5", True, "轨迹已报告（Psi(t) 采样）")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
ax.plot(H0, psi_o, "o-", label="lambda=0")
ax.plot(H0, psi_1, "s-", label="lambda=1（登记符号）")
ax.set_xlabel("h0（写入幅度）"); ax.set_ylabel("Psi")
ax.set_title("(a) 交错序参量对写入幅度")
ax.legend(fontsize=8)
ax = axs[1]
ax.plot(H0, [rr["R"] for _, lam, rr in rows if lam == 0.0], "o-", label="lambda=0")
ax.plot(H0, [rr["R"] for _, lam, rr in rows if lam == 1.0], "s-", label="lambda=1")
ax.set_xlabel("h0"); ax.set_ylabel("R（接力连续性）")
ax.set_title("(b) 接力连续性")
ax.legend(fontsize=8)
ax = axs[2]
ts = [x[0] for x in traj]
ax.plot(ts, [x[1]["psi"] for x in traj], "-", label="Psi")
ax.plot(ts, [x[1]["R"] for x in traj], "--", label="R")
ax.set_xlabel("步"); ax.set_ylabel("值")
ax.set_title("(c) 轨迹（h0=1、lambda=1）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_w1_staggered.png", dpi=140)
print("  图已存 images/bh7_w1_staggered.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
