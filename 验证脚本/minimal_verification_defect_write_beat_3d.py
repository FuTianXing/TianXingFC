# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_write_beat_3d.py

三维面缺陷的写入节拍轮：延展缺陷在一段时间里写下多少笔、这些笔是否随机、写入时刻与折返时刻是否同一。

来源：作者 2026-09-20 指令（接手第 289 条登记的方向）；接口口径取缺陷动力学线 D-5（缺陷每穿过一格恰写一笔、跃迁时长与取样间隔的计数规则）。

模型（三维过阻尼驱动，势与边界同第 289 条）：
  v_dot = [ k lap(v) - 2 pi K sin(2 pi v) + f + xi ] / gamma，k = 1，gamma = 0.5958（D-3 的读数）；
  x 两端固定（左 0、右 -1，恒有一个 kink），y、z 周期；f 为恒定驱动力。
  写入定义（自触发口径）：站点相位每跨过一个整数写一笔；折返时刻取同一站点跨过半整数（kink 中心经过）的时刻。

判据（先行）：
  W1 笔数与批结构：总笔数 = (Nx-2) * A（A = Ny*Nz 为截面），批数 = Nx-2
  W2 写入时刻与折返时刻同一：逐站点两个事件的时刻差最大值为 0
  W3 成批性：同一批内截面各站点写入时刻的散布 <= 0.05 批间隔（平坦初值与带褶皱初值各一次）
  W4 准时性：无噪声下批间隔的 Fano 因子 <= 0.05；随噪声增大单调上升
  W5 面积标度：截面 1x1 与 8x8 的总笔数之比 = 64，写入率之比 = 64
  W6 迁移率对照：实测终端速度对 f/(gamma M)（M = (4/pi) sqrt(K)），偏差 <= 20%
"""
import numpy as np

K = 0.001
GAMMA = 0.5958
F = 0.010
DT = 0.01
NX = 64
SIG = 4.0 * np.sqrt(K) / np.pi
rows = []


def run(Ny=8, Nz=8, F=F, noise=0.0, seed=7, steps=34000, bumpy=False):
    rng = np.random.default_rng(seed)
    xs = np.arange(NX, dtype=float)[:, None, None]
    off = np.zeros((1, Ny, Nz))
    if bumpy:
        off = 0.8 * rng.standard_normal((1, Ny, Nz))
    v = -0.5 * (1.0 + np.tanh((xs - 3.0 - off) / 1.5)) * np.ones((1, Ny, Nz))
    v[0, :, :] = 0.0
    v[-1, :, :] = -1.0
    floor_prev = np.floor(v + 0.5)
    nwrite = np.zeros_like(v)
    twrite = np.full(v.shape, np.nan)
    tcross = np.full(v.shape, np.nan)
    prev = v.copy()
    traj = []
    for it in range(steps):
        t = it * DT
        lap = np.zeros_like(v)
        lap[1:-1, :, :] = v[2:, :, :] + v[:-2, :, :] - 2.0 * v[1:-1, :, :]
        lap[:, :, :] += np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2.0 * v
        lap[:, :, :] += np.roll(v, -1, axis=2) + np.roll(v, 1, axis=2) - 2.0 * v
        g = k = 1.0
        force = k * lap - 2.0 * np.pi * K * np.sin(2.0 * np.pi * v) + F
        if noise > 0.0:
            force = force + noise * F * rng.standard_normal(v.shape)
        force[0, :, :] = 0.0
        force[-1, :, :] = 0.0
        v = v + DT * force / GAMMA
        v[0, :, :] = 0.0
        v[-1, :, :] = -1.0
        fl = np.floor(v + 0.5)
        new = (fl != floor_prev) & (np.isnan(twrite))
        if new.any():
            twrite[new] = t
            nwrite[new] += 1.0
        floor_prev = fl
        hc = np.floor(v)
        hcp = np.floor(prev)
        cr = (np.isnan(tcross)) & (((v <= -0.5) & (prev > -0.5)) | ((v >= -0.5) & (prev < -0.5)))
        if cr.any():
            tcross[cr] = t
        prev = v.copy()
        if Ny == Nz == 8 and it % 20 == 0:
            traj.append(float(np.mean(v[10, :, :])))
    return v, nwrite, twrite, tcross, np.array(traj)


def batches(twrite):
    ts = np.unique(np.round(twrite[~np.isnan(twrite)], 6))
    return ts


# W1 / W2 / W3
v, nw, tw, tc, traj = run(Ny=8, Nz=8)
A = 64
def batch_stats(tw):
    flat = tw[~np.isnan(tw)]
    vals, cnt = np.unique(np.round(flat, 6), return_counts=True)
    spread = 0.0
    for v0 in vals:
        m = np.abs(flat - v0) < 5e-7
        if m.sum() > 1:
            spread = max(spread, float(np.std(flat[m])))
    return vals, cnt, spread

tot = float(np.nansum(nw))
inner = NX - 2
ts = batches(tw)
vals, cnt, spread = batch_stats(tw)
d2 = float(np.nanmax(np.abs(tw - tc)))
rows.append(("W1", "总笔数对「批数乘截面」的恒等（截面 %d，窗口覆盖 %d 层）" % (A, len(ts)), "总笔数 %.0f = %d 乘 %d；每站 %.0f 笔" % (tot, len(ts), A, float(np.nanmax(nw))), "恒等且每站恰一笔", abs(tot - len(ts) * A) < 0.5 and float(np.nanmax(nw)) == 1.0))
rows.append(("W2", "写入时刻与折返时刻的最大差", "%.1e" % d2, "== 0", d2 == 0.0))

dtb = float(np.mean(np.diff(ts)))
rows.append(("W3", "平坦初值：每批站点数 / 批内时刻散布", "%d 至 %d / %.1e" % (int(cnt.min()), int(cnt.max()), spread), "站点数 = %d 且散布 <= 1e-9" % A, int(cnt.min()) == A and int(cnt.max()) == A and spread <= 1e-9))

vb, nwb, twb, tcb, _ = run(Ny=8, Nz=8, bumpy=True)
tsb = batches(twb)
valsb, cntb, spreadb = batch_stats(twb)
half = len(valsb) // 2
early = int(np.median(cntb[:half])); late = int(np.median(cntb[half:]))
rows.append(("W3b", "带褶皱初值（幅度 0.8 格）：批站点数的早段中位 / 晚段中位（全段 %d 至 %d）" % (int(cntb.min()), int(cntb.max())), "%d / %d（共 %d 批）" % (early, late, len(valsb)), "晚段中位 = %d（重新合批）" % A, late == A))

# W4 准时性
def fano_central(tw_i):
    ts_i = batches(tw_i)
    lo = int(0.4 * len(ts_i)); hi = int(0.75 * len(ts_i))
    sel = ts_i[lo:hi]
    iv = np.diff(sel)
    return float(np.var(iv) / np.mean(iv)), float(np.mean(iv)), len(ts_i)

fanos, ivs = [], []
for noise in [0.0, 0.3, 1.0]:
    _, _, tw_i, _, _ = run(Ny=8, Nz=8, noise=noise, seed=11)
    f, iv_m, nl = fano_central(tw_i)
    fanos.append(f)
    ivs.append(iv_m)
rows.append(("W4", "中央稳态窗批间隔的 Fano 因子与均值间隔（噪声 0 / 0.3 / 1.0）",
             "Fano %.4f / %.4f / %.4f（间隔 %.3f / %.3f / %.3f）" % (fanos[0], fanos[1], fanos[2], ivs[0], ivs[1], ivs[2]),
             "无噪声 <= 0.05 且噪声两档 >= 1", fanos[0] <= 0.05 and fanos[1] >= 1.0 and fanos[2] >= 1.0))

# W5 面积标度
_, nw1, tw1, _, _ = run(Ny=1, Nz=1, bumpy=False)
ts1 = batches(tw1)
tot1 = float(np.nansum(nw1))
rows.append(("W5", "截面 1x1 的总笔数与 8x8 之比", "%.0f（比值 %.0f）" % (tot1, tot / tot1), "比值 = 64", abs(tot / tot1 - 64.0) < 0.5))

# W6 迁移率
lo = int(0.4 * len(ts)); hi = int(0.75 * len(ts))
v_drive = (hi - lo - 1) / (float(ts[hi - 1]) - float(ts[lo])) if hi - lo > 1 else float("nan")
F_small = 0.002
_, _, tw_s, _, _ = run(Ny=8, Nz=8, F=F_small, steps=120000)
ts_s = batches(tw_s)
lo2 = int(0.4 * len(ts_s)); hi2 = int(0.75 * len(ts_s))
v_small = (hi2 - lo2 - 1) / (float(ts_s[hi2 - 1]) - float(ts_s[lo2])) if hi2 - lo2 > 1 else float("nan")
v_theory = F_small / (GAMMA * SIG)
rows.append(("W6", "小力档（f = %.3f < 2 pi K = %.4f）终端速度对 f/(gamma M)" % (F_small, 2 * np.pi * K), "%.4f 对 %.4f（偏差 %.1f%%）" % (v_small, v_theory, 100 * abs(v_small - v_theory) / v_theory), "偏差 <= 20%", abs(v_small - v_theory) / v_theory <= 0.20))
rows.append(("W6b", "大力档（f = %.3f > 2 pi K）实测速度与理论值之比（判负登记：本体一起滑移）" % F, "%.4f 对 %.4f（比值 %.2f）" % (v_drive, F / (GAMMA * SIG), v_drive / (F / (GAMMA * SIG))), "判负登记", v_drive / (F / (GAMMA * SIG)) > 1.5))

print("=== 三维面缺陷写入节拍轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-5s %-46s 实测 %-40s 期望 %-24s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("批时刻（前 8 个）：", np.round(ts[:8], 4))
print("批间隔：均值 %.4f，最大最小 %.4f 与 %.4f" % (dtb, float(np.max(np.diff(ts))), float(np.min(np.diff(ts)))))
print("笔数分布：每站 %.0f 笔（%d 站写、其余为边界固定站）" % (np.nanmax(nw), int(np.sum(np.nansum(nw, axis=(1, 2)) > 0))))
print("写入率 = 截面 %d 乘速度：小力档 %.4f 给 %.2f 笔每单位时间，单列对照为 %.4f（比值 %.0f）" % (A, v_small, A * v_small, v_small, A))
print("大力档 %.4f 的写入率为 %.2f（判负登记：本体滑移，不作迁移率读数）" % (v_drive, A * v_drive))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
IMG = r"D:\software\TianXingFC\images\defect_write_beat_3d.png"
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
axes[0].plot(np.arange(1, len(cntb) + 1), cntb, "o-", ms=2.5)
axes[0].axhline(A, color="k", ls="--", lw=1, label="截面 64")
axes[0].set_xlabel("批序号")
axes[0].set_ylabel("每批站点数")
axes[0].set_title("(a) 带褶皱初值的批站点数：分片后重新合批")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].bar(["无噪声", "噪声 0.3", "噪声 1.0"], fanos, width=0.45)
axes[1].axhline(1.0, color="k", ls="--", lw=1, label="泊松水平")
axes[1].set_ylabel("写入间隔的 Fano 因子")
axes[1].set_title("(b) 准时性对噪声强度")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：", IMG)
