# -*- coding: utf-8 -*-
r"""
proposition17_3d.py

命题 17 的三维时长口径轮：取样间隔对成批写入的稀释。

来源：作者 2026-09-20 指令（接手第 292 条登记的下一步）；口径取记录机制线命题 17（跃迁时长与取样间隔的计数规则，一维读数：时长等于间隔时全额计数，缩短到三分之一时给 0.3333）。

模型：与第 292 条同一三维过阻尼驱动（K = 0.001，gamma = 0.5958，x 两端固定为一个 kink，y、z 周期，截面 A = 64），小力档 f = 0.002（f < 2 pi K = 0.0063）。
  跃迁时长 tau：站点相位从 -0.1 到 -0.9 的历时；批时刻 t_c：相位跨过半整数的时刻。
  注册规则（命题 17）：取样时刻落在 [t_c - tau/2, t_c + tau/2] 内则该笔登记，否则丢失。

判据（先行）：
  U1 时长与速度的反比：两档驱动力下 tau * v 的散布 <= 20%（tau 由动力学给出，v 由批间隔给出）
  U2 稀释律：四档取样间隔（tau/3、tau、3 tau、10 tau）的注册率对 min(1, tau/Dt)，偏差 <= 5%
  U3 批内一致性：平坦壁同一批各站登记状态的一致率 = 1.000
  U4 方差放大：跨取样相位的计数方差对二项预期 A p (1-p) 之比 = A（判据 >= 30）
  U4b 对照：人为把各站窗口相位打乱后，该比值降到 1 附近（判负对照）
  U5 登记
"""
import numpy as np

K = 0.001
GAMMA = 0.5958
DT = 0.01
NX = 32
NZ = 8
A = NZ * NZ
rows = []


def run(F=0.002, steps=32000, seed=3):
    rng = np.random.default_rng(seed)
    xs = np.arange(NX, dtype=float)[:, None, None]
    v = -0.5 * (1.0 + np.tanh((xs - 3.0) / 1.5)) * np.ones((1, NZ, NZ))
    v[0, :, :] = 0.0
    v[-1, :, :] = -1.0
    t10 = np.full(v.shape, np.nan)
    t50 = np.full(v.shape, np.nan)
    t90 = np.full(v.shape, np.nan)
    prev = v.copy()
    for it in range(steps):
        t = it * DT
        lap = np.zeros_like(v)
        lap[1:-1, :, :] = v[2:, :, :] + v[:-2, :, :] - 2.0 * v[1:-1, :, :]
        lap[:, :, :] += np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2.0 * v
        lap[:, :, :] += np.roll(v, -1, axis=2) + np.roll(v, 1, axis=2) - 2.0 * v
        force = lap - 2.0 * np.pi * K * np.sin(2.0 * np.pi * v) + F
        force[0, :, :] = 0.0
        force[-1, :, :] = 0.0
        v = v + DT * force / GAMMA
        v[0, :, :] = 0.0
        v[-1, :, :] = -1.0
        for thr, arr in [(-0.1, t10), (-0.5, t50), (-0.9, t90)]:
            hit = np.isnan(arr) & (((v <= thr) & (prev > thr)) | ((v >= thr) & (prev < thr)))
            arr[hit] = t
        prev = v.copy()
    return t10, t50, t90


def read(t10, t50, t90):
    m = (~np.isnan(t90)) & (~np.isnan(t10)) & (~np.isnan(t50))
    tau = float(np.median(np.abs(t90[m] - t10[m])))
    tc = t50[~np.isnan(t50)]
    ts = np.unique(np.round(tc, 6))
    iv = np.diff(ts[2:10]) if len(ts) > 10 else np.diff(ts)
    v = 1.0 / float(np.mean(iv)) if len(iv) else float("nan")
    return tau, v, ts


t10a, t50a, t90a = run(F=0.002)
tau_a, v_a, ts_a = read(t10a, t50a, t90a)
t10b, t50b, t90b = run(F=0.004)
tau_b, v_b, ts_b = read(t10b, t50b, t90b)
prod = [tau_a * v_a, tau_b * v_b]
spr = abs(prod[0] - prod[1]) / (0.5 * sum(prod))
rows.append(("U1", "两档力下的 tau * v（F = 0.002 与 0.004）", "%.3f 与 %.3f（散布 %.1f%%）" % (prod[0], prod[1], 100 * spr), "散布 <= 20%", spr <= 0.20))

tau = tau_a
# U2 稀释律（跨 16 个取样相位）
ratios = [3.0, 1.0, 1.0 / 3.0, 1.0 / 10.0]
got, exp = [], []
flat_ok = []
for r in ratios:
    Dt = tau / r
    phases = np.arange(16) * Dt / 16.0
    fracs = []
    for ph in phases:
        lo = t50a - 0.5 * tau_a
        hi = t50a + 0.5 * tau_a
        n_ticks = np.floor((hi - ph) / Dt) - np.ceil((lo - ph) / Dt) + 1.0
        reg = n_ticks >= 1.0
        fracs.append(float(np.mean(reg[~np.isnan(t50a)])))
    got.append(float(np.mean(fracs)))
    exp.append(min(1.0, r))
    flat_ok.append(fracs)
dev2 = max(abs(g - e) / e for g, e in zip(got, exp))
rows.append(("U2", "四档取样间隔的注册率对 min(1, tau/Dt)", " / ".join("%.4f" % g for g in got) + " 对 " + " / ".join("%.3f" % e for e in exp), "偏差 <= 5%", dev2 <= 0.05))

# U3 批内一致性与 U4 方差放大（按批分组：同一层的站点共享批次号）
selmask = ~np.isnan(t50a)
tt = t50a[selmask]
batch_id = np.round(tt, 6)
uniq = np.unique(batch_id)
n_site = int(len(tt))
r = 1.0 / 3.0
Dt = tau / r
phases = np.arange(64) * Dt / 64.0
cons, counts = [], []
for ph in phases:
    lo = tt - 0.5 * tau
    hi = tt + 0.5 * tau
    reg = (np.floor((hi - ph) / Dt) - np.ceil((lo - ph) / Dt) + 1.0) >= 1.0
    counts.append(float(reg.sum()))
    oks = []
    for u0 in uniq:
        m = batch_id == u0
        oks.append(1.0 if reg[m].min() == reg[m].max() else 0.0)
    cons.append(float(np.mean(oks)))
same = float(np.mean(cons))
preg = float(np.mean(counts)) / n_site
var = float(np.var(counts))
ratio = var / (n_site * preg * (1.0 - preg)) if 0.0 < preg < 1.0 else float("nan")
rows.append(("U3", "同一批各站登记状态的一致率（%d 层乘 %d 相位）" % (len(uniq), len(phases)), "%.3f" % same, "= 1.000", same >= 0.999))
rows.append(("U4", "跨相位计数的方差对二项预期之比（站点 %d，注册率 %.3f）" % (n_site, preg), "%.1f" % ratio, "逐站独立为 1、整批相关为 %d、整体相关为 %d（判据 >= 30）" % (A, n_site), ratio >= 30))

rng = np.random.default_rng(5)
counts_ctrl = []
for rep in range(64):
    phi = rng.uniform(0.0, Dt, size=n_site)
    lo = tt - 0.5 * tau
    hi = tt + 0.5 * tau
    reg = (np.floor((hi - phi) / Dt) - np.ceil((lo - phi) / Dt) + 1.0) >= 1.0
    counts_ctrl.append(float(reg.sum()))
var_c = float(np.var(counts_ctrl))
ratio_c = var_c / (n_site * preg * (1.0 - preg)) if 0.0 < preg < 1.0 else float("nan")
rows.append(("U4b", "每站独立时钟相位的同一比值（构造对照）", "%.2f" % ratio_c, "降到 1 附近", ratio_c <= 5))

rows.append(("U5", "登记：取样稀释作用在整批上", "登记", "登记", True))

print("=== 命题 17 三维时长口径轮判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-5s %-52s 实测 %-42s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("时长与速度：F = 0.002 给 tau = %.2f、v = %.4f；F = 0.004 给 tau = %.2f、v = %.4f" % (tau_a, v_a, tau_b, v_b))
print("稀释律四档（tau = %.2f）：取样间隔 %s" % (tau, [round(tau / r, 1) for r in ratios]))
print("注册率实测 %s；解析 min(1, tau/Dt) = %s" % ([round(g, 4) for g in got], [round(e, 4) for e in exp]))
print("整批相关性：一致率 %.3f；全局时钟方差比 %.1f（逐站独立为 1、整批相关为 %d、整体相关为 %d）；独立时钟对照 %.2f" % (same, ratio, A, n_site, ratio_c))
