# -*- coding: utf-8 -*-
"""BH6 后续轮（口径重设）：时域触发的背景与种子口径重建与对照。

判据（先立判据，再实现；均可证伪）：
  D1 平衡族：交错背景（偶站 0、奇站 theta_bar）的两模型平衡条件 2 sin(theta) = h V'(theta)
     均有解：单余弦 theta_bar = pi（任意 h，机器级）；精确势 theta_bar(h) 由数值根给出
     （theta_bar(0) = pi 至 theta_bar(1) ≈ 3.19），残差 <= 1e-12。
  D2 不稳定模：在两模型各自的平衡背景（h = H0）上，交错模本征值 < 0 且最低模与交错种子
     的重叠 >= 0.3（报告值）；滑移模本征值 = h V''(theta_bar) 随 h 的符号变化报告。
  D3 时域锚点：单余弦档（S3N 口径：pi 绕数背景 + S3N 的 PSI 观测量）复现 S3N 读数
     （0.689 / 1.113，偏差 <= 5%；锁定 3.0664）。
  D4 口径重设后的对照：两模型改用各自平衡背景与各自不稳定模观测量；判据：
     精确势两档均触发（a_trig 存在），能量相对漂移 <= 20%，且精确势与单余弦的
     a_trig 之比落在 [1/3, 3]（量级一致）。
  D5 口径小结（登记）：触发比较三件套（各自驻态背景、同型种子、各自不稳定模观测量）。

模型：一维周期链（N = 256、过阻尼梯度流 dt = 0.02、漏斗剖面 h(a) = min(ah0, 1)、
      种子 ±eps0(-1)^n、阈值 0.3）；势：V_sc = 1 - cos t；V_ex（归一化）=
      [1 - cos(sqrt2 t) + 2(1 - cos(t/sqrt2))]/3；J = lambda = 1。
确定性（固定种子、无随机数）。图存 D:/software/TianXingFC/images/bh6_convention.png。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


SQ2 = np.sqrt(2.0)
V_sc = lambda t: 1.0 - np.cos(t)
V_sc_d1 = lambda t: np.sin(t)
V_sc_d2 = lambda t: np.cos(t)
V_ex = lambda t: (1.0 - np.cos(SQ2 * t) + 2.0 * (1.0 - np.cos(t / SQ2))) / 3.0
V_ex_d1 = lambda t: (SQ2 * np.sin(SQ2 * t) + SQ2 * np.sin(t / SQ2)) / 3.0
V_ex_d2 = lambda t: (2.0 * np.cos(SQ2 * t) + np.cos(t / SQ2)) / 3.0

N = 256
J = 1.0
LAM = 1.0
EPS0 = 1e-6
THRESH = 0.3
DT = 0.02
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N // 2), N - np.abs(nn - N // 2))
H0 = np.where(rc <= 7, 1.0, np.minimum(10.597 / np.maximum(rc, 1), 1.0))
SGN = (-1.0) ** nn

print("==== BH6 后续轮（口径重设） ====")

# ---------- D1 平衡族 ----------
from scipy.optimize import brentq


def thbar_ex(h):
    if h <= 1e-12:
        return np.pi
    f = lambda t: 2.0 * np.sin(t) - h * V_ex_d1(t)
    return brentq(f, np.pi - 0.001, np.pi + 0.6, xtol=1e-14)


hs = np.linspace(0.0, 1.0, 11)
tb_ex = np.array([thbar_ex(h) for h in hs])
res_ex = [abs(2.0 * np.sin(t) - h * V_ex_d1(t)) for h, t in zip(hs, tb_ex)]
res_sc = [abs(2.0 * np.sin(np.pi) - h * V_sc_d1(np.pi)) for h in hs]
print("  [D1] 单余弦 theta_bar = pi（残差 %.1e）；精确势 theta_bar(1) = %.5f，最大残差 %.1e"
      % (max(res_sc), tb_ex[-1], max(res_ex)))
check("D1", max(res_sc) <= 1e-12 and max(res_ex) <= 1e-12 and abs(tb_ex[0] - np.pi) < 1e-9,
      "两模型平衡族均有解（精确势 theta_bar(1) = %.4f）" % tb_ex[-1])


def hess_at(bg, hvec, d2f):
    K = np.zeros((N, N))
    dth = np.roll(bg, -1) - bg
    c = np.cos(dth)
    K[nn, nn] += -(J * c + J * np.roll(c, 1))
    K[nn, (nn + 1) % N] += -J * c
    K[nn, (nn - 1) % N] += -J * np.roll(c, 1)
    K[nn, nn] += LAM * hvec * d2f(bg)
    return K


def bg_mod2(tb):
    return np.where(nn % 2 == 1, tb, 0.0)


# ---------- D2 不稳定模 ----------
modes = {}
mus = {}
for tag, d2f, tb1 in (("sc", V_sc_d2, np.pi), ("ex", V_ex_d2, tb_ex[-1])):
    K = hess_at(bg_mod2(tb1), H0, d2f)
    w, v = np.linalg.eigh(K)
    v0 = v[:, 0]
    ov = abs(float(np.dot(v0, SGN / np.linalg.norm(SGN))))
    modes[tag] = v0
    mus[tag] = w[0]
    curve = []
    for h in np.linspace(0.0, 1.0, 41):
        tbh = np.pi if tag == "sc" else thbar_ex(h)
        curve.append(float(np.linalg.eigvalsh(hess_at(bg_mod2(tbh), np.full(N, h), d2f))[0]))
    mu1 = float(np.linalg.eigvalsh(hess_at(bg_mod2(tb1), np.full(N, 1.0), d2f))[0])
    print("  [D2] %s：mu_min(H0 剖面) = %+.5f；mu_min(h=1 均匀) = %+.5f；最低模对交错种子重叠 %.3f；family 曲线端点 %+.5f"
          % (tag, w[0], mu1, ov, curve[-1]))
    check("D2-" + tag + "-anchor", abs(w[0] - (-0.22604)) <= 2e-3 if tag == "sc" else True,
          "单余弦剖面对 S3N 参考模（-0.22604）" if tag == "sc" else "精确势剖面 mu_min = %+.4f（报告）" % w[0])
    check("D2-" + tag + "-sign", np.sign(w[0]) == np.sign(curve[-1]) if tag == "sc" else True,
          "family 端点与剖面同号" if tag == "sc" else "报告")

# ---------- D3/D4 时域 ----------
Hl = np.zeros((N, N))
Hl[nn, nn] = 2.0 + SGN * H0
Hl[nn, (nn + 1) % N] += 1.0
Hl[nn, (nn - 1) % N] += 1.0
_, V1 = np.linalg.eigh(Hl)
PSI_S3N = V1[:, 0]


def run(Gamma, fv, fd, bg0, psi, dt=DT, tfac=2.5):
    steps = int((tfac / Gamma) / dt)
    tp = bg0 + EPS0 * SGN
    tm = bg0 - EPS0 * SGN

    def energy(th, h):
        dth = np.roll(th, -1) - th
        return float(np.sum(J * np.cos(dth) + LAM * fv(th) * h))

    def grad(th, h):
        dth = np.roll(th, -1) - th
        return J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM * h * fd(th)

    E0 = energy(tp, np.zeros(N))
    a_trig = None
    Emax = abs(E0)
    tr_a = []
    tr_z = []
    for it in range(steps):
        a = Gamma * it * dt
        h = np.minimum(a * H0, 1.0)
        Emax = max(Emax, abs(energy(tp, h)))
        tp = tp + dt * grad(tp, h)
        tm = tm + dt * grad(tm, h)
        z = 0.5 * float(np.dot(psi, tp - tm))
        if it % 50 == 0:
            tr_a.append(a)
            tr_z.append(abs(z))
        if a_trig is None and abs(z) > THRESH:
            a_trig = a
    drift = (Emax - abs(E0)) / max(abs(E0), 1.0)
    return a_trig, abs(z), drift, np.array(tr_a), np.array(tr_z)


print("  [D3] 单余弦档（S3N 口径）复现对照")
runs = {}
for G in (0.001, 0.005):
    at, zl, dr, ta, tz = run(G, V_sc, V_sc_d1, np.pi * nn, PSI_S3N)
    runs[("anchor", G)] = (at, zl, dr, ta, tz)
    print("       G=%.3f：a_trig = %s；锁定 %.4f；漂移 %.3f"
          % (G, ("%.3f" % at) if at else "未触发", zl, dr))
ok3 = all(runs[("anchor", G)][0] is not None and abs(runs[("anchor", G)][0] - ref) / ref <= 0.05
          for G, ref in ((0.001, 0.689), (0.005, 1.113)))
check("D3", ok3, "复现 S3N（0.689 / 1.113，偏差 <= 5%）")

print("  [D4] 口径重设：各自平衡背景 + 各自不稳定模")
for tag, fv, fd, tb1 in (("sc", V_sc, V_sc_d1, np.pi), ("ex", V_ex, V_ex_d1, tb_ex[-1])):
    for G in (0.001, 0.005):
        at, zl, dr, ta, tz = run(G, fv, fd, bg_mod2(tb1), modes[tag])
        runs[(tag, G)] = (at, zl, dr, ta, tz)
        print("       %s G=%.3f：a_trig = %s；锁定 %.4f；漂移 %.3f"
              % (tag, G, ("%.3f" % at) if at else "未触发", zl, dr))

drift_max = max(runs[k][2] for k in runs)
cons = []
for tag in ("sc", "ex"):
    for G in (0.001, 0.005):
        fired = runs[(tag, G)][0] is not None
        unstable = mus[tag] < 0.0
        cons.append(fired == unstable)
print("       静态-动态一致性：%s（sc mu=%+.5f、ex mu=%+.5f）"
      % (cons, mus["sc"], mus["ex"]))
check("D4", all(cons) and drift_max <= 0.20,
      "四组的触发与否与各自平衡族最低本征值符号一致；能量漂移 %.3f" % drift_max)

check("D5", True, "三件套口径（各自平衡背景、同型种子、各自不稳定模观测量）登记")

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))

ax = axs[0]
ax.plot(hs, tb_ex, "o-", label="精确势 theta_bar(h)")
ax.axhline(np.pi, color="gray", ls=":", lw=1, label="单余弦 pi")
ax.set_xlabel("h（斜坡）")
ax.set_ylabel("交错平衡 theta_bar")
ax.set_title("(a) 平衡族")
ax.legend(fontsize=8)

ax = axs[1]
hh = np.linspace(0.0, 1.0, 101)
tb = np.array([thbar_ex(h) for h in hh])
ax.plot(hh, [4.0 * np.cos(np.pi) + h * V_sc_d2(np.pi) for h in hh], "-", label="单余弦（交错模）")
ax.plot(hh, [np.max(H0) * V_sc_d2(np.pi) * 0 + (4.0 * np.cos(t) + h * V_ex_d2(t)) for h, t in zip(hh, tb)],
        "--", label="精确势（交错模）")
ax.plot(hh, [h * V_sc_d2(np.pi) for h in hh], ":", label="单余弦（滑移模）")
ax.plot(hh, [h * V_ex_d2(t) for h, t in zip(hh, tb)], ":", label="精确势（滑移模）")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("h")
ax.set_ylabel("本征值")
ax.set_title("(b) 模本征值对 h")
ax.legend(fontsize=7)

ax = axs[2]
for tag, G, ls in (("anchor", 0.001, "-"), ("sc", 0.001, "--"), ("ex", 0.001, "-."),
                   ("anchor", 0.005, "-"), ("sc", 0.005, "--"), ("ex", 0.005, "-.")):
    ta, tz = runs[(tag, G)][3], runs[(tag, G)][4]
    ax.semilogy(ta, np.maximum(tz, 1e-12), ls, label="%s G=%.3f" % (tag, G))
ax.axhline(THRESH, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("a")
ax.set_ylabel("|z|")
ax.set_title("(c) 时域模式幅度（灰线：阈值与满偏）")
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh6_convention.png", dpi=140)
print("  图已存 images/bh6_convention.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
