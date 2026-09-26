# -*- coding: utf-8 -*-
"""口径链判定：力型无关性与缺陷几何的浴维数（第 432 条，候选丁段三判定轮）。

判据：
  A 组（力型，3D 浴固定，点缺陷）：
    A1 单方向梯度、A2 平面梯度、A3 全键均匀 -> gamma_F 均应约 3/2（力图像维数不占指数位）
  B 组（线缺陷几何，3D 浴）：沿 z 的相干因子 sinc^2(L kz/2)（精确式，含 sin(kz/2) 分母），
    L = 1, 4, 16, 64 -> 尾指数从 3/2 向 1 过渡的形态（是否出现稳定中间平台）
  C 组（对照）：2D 浴 + 单方向梯度 -> gamma_F 约 1（B 组 L 无穷的极限读法）

方法：同第 425 条的蒙特卡洛 k 积分（均匀采样 BZ、同点二时间关联、对数分段 RMS 包络、
      3 sigma 噪声地板、3 sigma 以上段双对数拟合）；本轮把 7 个通道共享同批采样。
纪律：每档给两个窗口的斜率（跨窗稳定性检查，第七类陷阱）；跨窗翻动者登记为不可用。

运行：python 验证脚本/minimal_verification_defect_geometry_tail.py
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "defect_geometry_tail.png")


def make_ts(nseg, per, tmin, tmax):
    edges = np.logspace(math.log10(tmin), math.log10(tmax), nseg + 1)
    ts = np.concatenate([np.linspace(edges[i], edges[i + 1], per) for i in range(nseg)])
    return edges, ts


def mc_3d(M, nseg, per, tmin, tmax, seed, batch=200000):
    r = np.random.default_rng(20260926 + seed)
    edges, ts = make_ts(nseg, per, tmin, tmax)
    nt = len(ts)
    names = ["A1", "A2", "A3", "L1", "L4", "L16", "L64"]
    C = {n: np.zeros(nt) for n in names}
    C2 = {n: np.zeros(nt) for n in names}
    done = 0
    while done < M:
        mb = min(batch, M - done)
        ks = r.uniform(-np.pi, np.pi, (mb, 3))
        kx = ks[:, 0]; ky = ks[:, 1]; kz = ks[:, 2]
        s2 = np.sin(kx / 2.0) ** 2 + np.sin(ky / 2.0) ** 2 + np.sin(kz / 2.0) ** 2
        w = 2.0 * np.sqrt(s2)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(w > 1e-12, 1.0 / w, 0.0)
        wA1 = 2.0 * np.sin(kx / 2.0) ** 2 * inv
        wA2 = 2.0 * (np.sin(kx / 2.0) ** 2 + np.sin(ky / 2.0) ** 2) * inv
        wA3 = 0.5 * w
        cz = np.sin(kz / 2.0)
        ws = {"A1": wA1, "A2": wA2, "A3": wA3}
        for L in (1, 4, 16, 64):
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = np.where(np.abs(cz) > 1e-12, (np.sin(L * kz / 2.0) / (L * cz)) ** 2, 1.0)
            ws["L%d" % L] = fac * wA1
        coswt = np.cos(w[:, None] * ts[None, :])
        for n in names:
            c = ws[n][:, None] * coswt
            C[n] += c.sum(axis=0)
            C2[n] += (c ** 2).sum(axis=0)
        done += mb
    out = {}
    for n in names:
        c = C[n] / M; c2 = C2[n] / M
        se = np.sqrt(np.maximum(c2 - c * c, 0.0) / M)
        mids = []; rms = []; sig = []
        idx = 0
        for i in range(nseg):
            seg = c[idx:idx + per]; segse = se[idx:idx + per]
            mids.append(math.sqrt(edges[i] * edges[i + 1]))
            rms.append(float(np.sqrt(np.mean(seg ** 2))))
            sig.append(float(np.mean(segse)))
            idx += per
        out[n] = (np.array(mids), np.array(rms), np.array(sig))
    return out


def fitc(m, e, s, lo, hi):
    msk = (m >= lo) & (m <= hi) & (e > 3.0 * s)
    if np.sum(msk) < 6:
        return float("nan"), 0
    return float(np.polyfit(np.log(m[msk]), np.log(e[msk]), 1)[0]), int(np.sum(msk))


def report(name, m, e, s, twin=None):
    idx = np.where(e > 3.0 * s)[0]
    if len(idx) == 0:
        print("  %s: no usable window" % name)
        return float("nan")
    lo, hi = m[idx[0]], m[idx[-1]]
    a, n = fitc(m, e, s, lo, hi)
    line = "  %s: window [%.1f, %.1f] (%d segs), gamma = %.3f (%d pts)" % (name, lo, hi, len(idx), a, n)
    if twin is not None:
        b, nb = fitc(m, e, s, twin[0], twin[1])
        line += "; sub-window [%.0f, %.0f]: %.3f (%d pts)" % (twin[0], twin[1], b, nb)
    print(line)
    return a


print("== A/B 组（3D 浴，M = 3e6） ==")
res = mc_3d(3000000, 48, 10, 10.0, 3e4, seed=32)

aA1 = report("A1 single-axis", *res["A1"], twin=(30.0, 300.0))
aA2 = report("A2 planar    ", *res["A2"], twin=(30.0, 300.0))
aA3 = report("A3 all-bonds ", *res["A3"], twin=(30.0, 300.0))
aL = {}
for L in (1, 4, 16, 64):
    aL[L] = report("L=%-3d line   " % L, *res["L%d" % L], twin=(30.0, 300.0))
print()

print("== C 组（2D 浴对照，M = 3e6） ==")
def mc_2d(M, nseg, per, tmin, tmax, seed, batch=200000):
    r = np.random.default_rng(20260926 + seed)
    edges, ts = make_ts(nseg, per, tmin, tmax)
    nt = len(ts)
    C = np.zeros(nt); C2 = np.zeros(nt)
    done = 0
    while done < M:
        mb = min(batch, M - done)
        ks = r.uniform(-np.pi, np.pi, (mb, 2))
        kx = ks[:, 0]
        s2 = np.sin(kx / 2.0) ** 2 + np.sin(ks[:, 1] / 2.0) ** 2
        w = 2.0 * np.sqrt(s2)
        with np.errstate(divide="ignore", invalid="ignore"):
            wg = np.where(w > 1e-12, 2.0 * np.sin(kx / 2.0) ** 2 / w, 0.0)
        c = wg[:, None] * np.cos(w[:, None] * ts[None, :])
        C += c.sum(axis=0); C2 += (c ** 2).sum(axis=0)
        done += mb
    c = C / M; c2 = C2 / M
    se = np.sqrt(np.maximum(c2 - c * c, 0.0) / M)
    mids = []; rms = []; sig = []
    idx = 0
    for i in range(nseg):
        seg = c[idx:idx + per]; segse = se[idx:idx + per]
        mids.append(math.sqrt(edges[i] * edges[i + 1]))
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
        sig.append(float(np.mean(segse)))
        idx += per
    return np.array(mids), np.array(rms), np.array(sig)

m2, e2, s2c = mc_2d(3000000, 48, 10, 10.0, 3e4, seed=33)
aC = report("C  2D-point  ", m2, e2, s2c, twin=(30.0, 300.0))
print()

print("== 期望对照 ==")
print("  A1/A2/A3: 3/2 = 1.500 (image-dimension independence)")
print("  L family: from 1.5 (point) toward 1.0 (line); transition shape is the reading")
print("  C: 1.0 (2D point)")
print("  target range for gamma_F from the alpha band: [1.12, 1.20]")

print("== S4 线缺陷两段结构的解析核（高斯窗替代 sinc^2，闭式） ==")
print("  model: C(t) ~ t^-1 * Int dkz e^{-a kz^2} cos(alpha0 t kz^2), a = L^2/8, alpha0 = sqrt(2)/8")
print("  closed form: |Int| = sqrt(pi) (a^2 + (alpha0 t)^2)^(-1/4)")
print("  t << L^2: |Int| -> const, total tail 1.0;  t >> L^2: |Int| -> t^-1/2, total tail 1.5")
for L in (4, 16, 64):
    tc = L * L / math.sqrt(2.0)
    print("  L=%-4d transition center t* = L^2/sqrt(2) = %.1f" % (L, tc))
print("  判读：有限长线的尾为两段（早 1.0、晚 1.5），过渡中心 t* ~ L^2；无中间平台，故缺陷几何不能给出稳定的非整数指数")
print()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for n, lab in (("A1", "single-axis"), ("A2", "planar"), ("A3", "all-bonds")):
    m, e, s = res[n]
    ax.loglog(m, e, "o-", ms=2.5, lw=1, label=lab)
mm = np.array([m[0], m[-1]])
ax.loglog(mm, 3.0 * mm ** (-1.5), "k--", lw=1, alpha=0.6, label="t^-1.5 ref")
ax.set_xlabel("t"); ax.set_ylabel("envelope RMS"); ax.set_title("A: force type (3D bath)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = axes[1]
for L in (1, 4, 16, 64):
    m, e, s = res["L%d" % L]
    ax.loglog(m, e, "o-", ms=2.5, lw=1, label="L=%d" % L)
ax.loglog(m2, e2, "s-", ms=2.5, lw=1, color="0.4", label="2D point (C)")
ax.loglog(mm, 5.0 * mm ** (-1.0), "k--", lw=1, alpha=0.6, label="t^-1 ref")
ax.set_xlabel("t"); ax.set_ylabel("envelope RMS"); ax.set_title("B: defect geometry (3D bath)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
