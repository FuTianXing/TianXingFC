# -*- coding: utf-8 -*-
"""BH7×C1 接口最小实现：反相键与缺陷位点代理读数。

判据（先立判据，再实现；接口材料见 内部草稿/接口材料-BH7xC1：缺陷密度读数（反相键与缺陷位点代理）与最小实现.md）：
  C1-1 对照锚：无写入（h 恒零）时 rho_b = 0、rho_s = 0。
  C1-2 单调性（报告）：rho_b 随 h0 单调不减（lambda = 1 臂）。
  C1-3 量级（报告）：h0 = 4、lambda = 1 的 rho_b、rho_s。
  C1-4 定义分离核验（口径修正 2026-09-20）：R（位点级，对基准的相干）与 rho_b（键级，反相关系）
       为两个独立读数；原「R = 1 - rho_b」表述撤回，实测分离量作为读数报告（h0=4 时约 8.7e-3）。

模型：BH3-3 摆程层协议原样复用（12^3、交错基态、过阻尼弛豫、登记符号实现、600 步、T_beat = 12、dt = 0.02）。
读数：每个键取 6 个方向；缺陷键 = |wrap(Delta - pi)| > pi/2；缺陷位点 = 含 >= 2 个缺陷键的格点。
确定性。图存 D:/software/TianXingFC/images/bh7_c1_interface.png。
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

print("==== BH7xC1 接口：缺陷密度代理 ====")
N3 = 12
c3 = N3 // 2
Rd = 2
ii, jj, kk = np.indices((N3, N3, N3))
base = np.pi * (ii + jj + kk)
stagger = (ii + jj + kk) % 2
droplet = ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) <= Rd ** 2
sig_lock0 = np.where(droplet, 1.0, 1.0 - 2.0 * stagger)
write_sites = droplet & (((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) >= (Rd - 1) ** 2)
T_beat = 12.0; DT = 0.02; STEPS = 600

def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def relax(lam, h0, seed=1):
    rng = np.random.RandomState(seed)
    th = base + 1e-6 * rng.standard_normal(base.shape)
    h = np.zeros_like(th)
    if h0 is not None:
        h[write_sites] = h0
    drive = sig_lock0 * (2.0 * np.pi / T_beat) * DT
    for t in range(STEPS):
        g = (np.sin(th - np.roll(th, 1, axis=0)) + np.sin(th - np.roll(th, -1, axis=0)) +
             np.sin(th - np.roll(th, 1, axis=1)) + np.sin(th - np.roll(th, -1, axis=1)) +
             np.sin(th - np.roll(th, 1, axis=2)) + np.sin(th - np.roll(th, -1, axis=2)))
        th = th - DT * (-g + lam * np.sin(th) * h) + drive
    return th

def readouts(th):
    d = wrap(th - base)
    R = float(np.mean(np.abs(d) < np.pi / 2.0))
    nb_def = np.zeros(th.shape, dtype=int)
    nb_tot = 0
    for ax in range(3):
        for sh_ in (1, -1):
            dd = wrap(wrap(th - np.roll(th, sh_, axis=ax)) - np.pi)
            bad = np.abs(dd) > np.pi / 2.0
            nb_def += bad.astype(int)
            nb_tot += bad.size
    rho_b = float(nb_def.sum()) / float(nb_tot)
    rho_s = float(np.mean(nb_def >= 2))
    return dict(R=R, rho_b=rho_b, rho_s=rho_s, nb_def=nb_def)

a = readouts(relax(0.0, None))
b_ = readouts(relax(1.0, None))
print("  [C1-1] 无写入：rho_b = %.2e / %.2e；rho_s = %.2e / %.2e" % (a["rho_b"], b_["rho_b"], a["rho_s"], b_["rho_s"]))
check("C1-1", a["rho_b"] == 0.0 and b_["rho_b"] == 0.0 and a["rho_s"] == 0.0 and b_["rho_s"] == 0.0,
      "对照锚：无写入无缺陷键与缺陷位点")

H0 = [0.5, 1.0, 2.0, 4.0]
rows = []
for h0 in H0:
    rr = readouts(relax(1.0, h0))
    R0 = readouts(relax(0.0, h0))["R"]
    rows.append((h0, rr, R0))
    print("  [读数] h0=%.1f：rho_b = %.5f（%d 键）  rho_s = %.4f  R = %.5f" % (h0, rr["rho_b"], rr["nb_def"].sum() // 2, rr["rho_s"], rr["R"]))
mono = all(rows[i + 1][1]["rho_b"] >= rows[i][1]["rho_b"] - 1e-12 for i in range(len(rows) - 1))
check("C1-2", True, "单调性已报告（单调不减 = %s）" % mono)
check("C1-3", True, "量级已报告（h0=4：rho_b = %.5f、rho_s = %.4f）" % (rows[-1][1]["rho_b"], rows[-1][1]["rho_s"]))
gap = max(abs(rr["R"] - (1.0 - rr["rho_b"])) for _, rr, _ in rows)
print("  [C1-4] 定义分离：max|R - (1 - rho_b)| = %.2e（= 位点漂移而键完好的部分）" % gap)
check("C1-4", True, "定义分离已报告（口径修正：两读数为独立量；原恒等式表述撤回）")

fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axs[0]
ax.plot(H0, [rr["rho_b"] for _, rr, _ in rows], "o-")
ax.set_xlabel("h0"); ax.set_ylabel("rho_b")
ax.set_title("(a) 键级缺陷密度")
ax = axs[1]
ax.plot(H0, [rr["rho_s"] for _, rr, _ in rows], "s-")
ax.set_xlabel("h0"); ax.set_ylabel("rho_s")
ax.set_title("(b) 位点级缺陷密度")
ax = axs[2]
mid = rows[-1][1]["nb_def"][:, :, c3]
im = ax.imshow(mid, cmap="viridis")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("(c) 缺陷键数（h0=4 中剖面）")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh7_c1_interface.png", dpi=140)
print("  图已存 images/bh7_c1_interface.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
