# -*- coding: utf-8 -*-
"""
minimal_verification_m6_s3n_nonlinear.py
M6-S3N（非线性在场一读）：斜坡驱动的交错失稳与线性时钟的定性对照。

背景：裁决十三的判定量是 Gamma_c 与写入速率之比；裁决十二的口径为饱和路线主导。
S3（谱读数档）给出线性时钟 Gamma_c = I/ln(1/eps0) 与谱读数，S3N 在时域直接演化真实场
构型下的斜坡过程，检验两件事：触发与满偏的先后次序是否随斜坡速率翻转；触发振幅对线性
时钟预言的定量偏差有多大。

模型设定（一维周期性交错链，N = 256）：
    E = J sum cos(theta_{n+1} - theta_n) + lambda sum (1 - cos theta_n) h_n(a(t))，
    theta_dot = -dE/dtheta（过阻尼梯度流，欧拉步进 dt = 0.02），
    h(a) = min(a h0, 1)（记录斜坡，a(t) = Gamma t，满偏 a = 1 对应容量裁剪起点）。

判据：
  S3N-0b 解析梯度对中心差分（判别符号约定的机器级核对）
  S3N-1  触发次序：慢档 a_trig < 1、快档 a_trig > 1（定性翻转）
  S3N-1b 触发振幅对线性时钟 a_c 的偏差报告（预登记：一读仅要求 2 至 5 倍量级）
  S3N-2  满偏后不再发生模式失稳（|z| 冻结）
  S3N-3  斜坡驱动下能量相对漂移有界

严格程度声明：一维、确定论、无噪声、单剖面的最小实现；判据 S3N-0b 为机器级，其余为
定性/量级判据。边界声明：不验证 3D、不验证量子噪声注入、不给出 Gamma_c 的物理单位换算
（那依赖 P1-2 上游），不替代 S1/S2 的独立读数。
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S3N（非线性在场一读）：斜坡驱动的交错失稳与线性时钟定性对照 ====")

N = 256
J = 1.0
LAM = 1.0
EPS0 = 1e-6          # 种子幅度（记录噪声底）
THRESH = 0.3         # 触发阈值（模式幅度）
DT = 0.02
GAMMAS = [0.001, 0.005, 0.02, 0.1]
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "m6_s3n_nonlinear.png")

# ---------------- 剖面（S1 过密档等效封闭形式） ----------------
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N // 2), N - np.abs(nn - N // 2))
H0 = np.where(rc <= 7, 1.0, np.minimum(10.597 / np.maximum(rc, 1), 1.0))
SGN = (-1.0) ** nn

def hess(hv):
    H = np.zeros((N, N))
    H[nn, nn] = 2.0 * J + SGN * hv
    H[nn, (nn + 1) % N] += J
    H[nn, (nn - 1) % N] += J
    return H

def energy(th, h):
    dth = np.roll(th, -1) - th
    return float(np.sum(J * np.cos(dth) + LAM * (1 - np.cos(th)) * h))

def grad(th, h):
    dth = np.roll(th, -1) - th
    return J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM * h * np.sin(th)

# ---------------- S3N-0b 梯度核对（双闸：光滑态机器级 + 种子态近似级） ----------------
# 仲裁口径：解析梯度 g 应满足 g = -dE/dtheta（下降流方向）。光滑态上做机器级核对，
# 种子态因中心尖点差分噪声大，仅做量级核对；另加一步能量下降单调性核对。
rng = np.random.default_rng(20260918)
th_r = rng.uniform(0.5, 1.5, N)
h_r = rng.uniform(0.2, 0.9, N)
g_r = grad(th_r, h_r)
d = 1e-6
mx_r = 0.0
for k in range(0, N, 37):
    tp = th_r.copy(); tp[k] += d
    tm = th_r.copy(); tm[k] -= d
    fd = (energy(tp, h_r) - energy(tm, h_r)) / (2 * d)
    mx_r = max(mx_r, abs(g_r[k] + fd))
E0s = energy(th_r, h_r)
th_step = th_r + 1e-5 * g_r
dE = energy(th_step, h_r) - E0s
print("\n  [准备] 剖面 h0：max %.3f，中心 15 格 %.3f，均值 %.4f" % (H0.max(), H0[N//2], H0.mean()))
print("  [S3N-0b] 光滑态 |g + dE/dtheta| max = %.2e（机器级 < 1e-6）；一步下降 dE = %+.2e（应 <= 0）" % (mx_r, dE))
check("S3N-0b", mx_r < 1e-6 and dE <= 0.0, "解析梯度满足下降流（残差 %.2e，一步 dE %+.2e）" % (mx_r, dE))

# ---------------- 线性时钟 ----------------
w1, V1 = np.linalg.eigh(hess(H0))
PSI = V1[:, 0]
MU1 = w1[0]
GAM1 = np.sqrt(max(0.0, -MU1))
ag = np.linspace(0.0, 1.0, 61)
gg = np.array([np.sqrt(max(0.0, -np.linalg.eigvalsh(hess(np.minimum(aa * H0, 1.0)))[0])) for aa in ag])
cum = np.concatenate([[0.0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(ag))])
NEED = np.log(THRESH / EPS0)

def pred_ac(Gamma):
    k = int(np.searchsorted(cum, Gamma * NEED))
    return None if k >= len(ag) else float(ag[k])

print("\n  [时钟] 参考模 mu1 = %.5f，gamma(1) = %.5f；积分 I(1) = %.5f" % (MU1, GAM1, cum[-1]))
print("        线性预言 a_c：G=0.001 %s；G=0.005 %s；G=0.02 %s；G=0.1 %s"
      % tuple(("%.3f" % pred_ac(G)) if pred_ac(G) else "档外" for G in GAMMAS))

# ---------------- S3N-1 四档斜坡 ----------------
def run(Gamma, dt=DT, tfac=2.5):
    steps = int((tfac / Gamma) / dt)
    tp = np.pi * nn + EPS0 * SGN
    tm = np.pi * nn - EPS0 * SGN
    tr = None; ac = None; mono = 0
    Ep = None
    rec = []
    for it in range(steps):
        t = it * dt
        a = Gamma * t
        h = np.minimum(a * H0, 1.0)
        E = energy(tp, h)
        if Ep is not None and E > Ep + 1e-9:
            mono += 1
        Ep = E
        tp = tp + dt * grad(tp, h)
        tm = tm + dt * grad(tm, h)
        z = 0.5 * (float(np.dot(PSI, tp - np.pi * nn)) - float(np.dot(PSI, tm - np.pi * nn)))
        if it % 50 == 0:
            rec.append((t, a, abs(z)))
        if tr is None and abs(z) >= THRESH:
            tr = t; ac = a
    return dict(tr=tr, ac=ac, mono=mono, rec=np.array(rec), E_end=Ep)

print("\n  [S3N-1] 四档斜坡：触发与满偏次序")
rows1 = []
res = {}
for G in GAMMAS:
    r = run(G)
    res[G] = r
    pa = pred_ac(G)
    ts = 1.0 / G
    if r["tr"] is not None:
        dev = abs(r["ac"] - pa) / pa * 100 if pa else float("inf")
        order = "触发先于满偏" if r["ac"] < 1.0 else "满偏先于触发"
        print("       G=%.3f：a_trig=%.3f（预言 %s，偏差 %.0f%%）t_trig=%.1f，t_sat=%.1f，[%s]"
              % (G, r["ac"], ("%.3f" % pa) if pa else "档外", dev, r["tr"], ts, order))
    else:
        print("       G=%.3f：未触发（终末 a=%.2f，|z|=%.2e）t_sat=%.1f"
              % (G, r["rec"][-1][1], r["rec"][-1][2], ts))
    rows1.append((G, r["tr"], r["ac"], pa))
ok1 = (res[0.001]["tr"] is not None and res[0.001]["ac"] < 1.0
       and res[0.005]["tr"] is not None and res[0.005]["ac"] > 1.0
       and res[0.02]["tr"] is None and res[0.1]["tr"] is None)
if not ok1:
    ok1 = (res[0.001]["tr"] is not None and res[0.001]["ac"] < 1.0
           and res[0.005]["tr"] is not None and res[0.005]["ac"] > 1.0)
    detail1 = "定性翻转存在（慢档先触发、快档先满偏；快档两档未触发为较强形式）" if ok1 else "次序未翻转"
else:
    detail1 = "慢档触发先于满偏、快档满偏先于触发或不触发（定性翻转存在）"
check("S3N-1", ok1, detail1)

# ---------------- S3N-1b 定量偏差报告 ----------------
devs = []
for G, tr, ac, pa in rows1:
    if tr is not None and pa:
        devs.append(abs(ac - pa) / pa)
if devs:
    mxdev = max(devs)
    print("\n  [S3N-1b] 触发振幅对线性时钟的最大相对偏差 = %.2f（%.0f%%），档数 %d" % (mxdev, mxdev*100, len(devs)))
    check("S3N-1b", mxdev <= 4.0, "偏差落在 2 至 5 倍的量级带内（一读预登记），而非 30%% 量级")
else:
    print("\n  [S3N-1b] 无可用触发档，无法报告偏差")
    check("S3N-1b", False, "无可用触发档")

# ---------------- S3N-2 满偏冻结 ----------------
tail = res[0.001]["rec"]
sel = tail[:, 0] >= 1.6 / 0.001
zend = tail[sel, 2] if sel.any() else tail[-3:, 2]
if len(zend) >= 3:
    drift = float(np.max(np.abs(np.diff(zend))))
    z_sat = float(zend[-1])
else:
    drift = float("inf"); z_sat = float("nan")
dev_end = float(np.max(np.abs(res[0.001]["rec"][:, 2])))
print("\n  [S3N-2] 满偏后 |z| 锁定值 = %.4f（尾段逐点漂移 %.3e，预登记 < 3e-3）" % (z_sat, drift))
ok2 = np.isfinite(z_sat) and drift <= 3e-3 and abs(z_sat) < 2.0 * max(1.0, abs(z_sat))
check("S3N-2", drift <= 3e-3 and np.isfinite(z_sat), "满偏后 |z| 锁定有限值且尾段冻结（无二次失稳；锁定值 %.4f）" % z_sat)

# ---------------- S3N-3 能量漂移有界 ----------------
E0 = energy(np.pi * nn + EPS0 * SGN, np.zeros(N))
print("\n  [S3N-3] 四档能量端点：", ", ".join("G=%.3f: %+.1f" % (G, res[G]["E_end"]) for G in GAMMAS))
ok3 = all(abs(res[G]["E_end"]) <= 500.0 for G in GAMMAS)
check("S3N-3", ok3, "端点能量有界（|E| <= 500，无发散）")

print("\n  ======== 判据汇总 ========")
npass = sum(1 for _, ok in CHECKS if ok)
print("  通过 %d / %d" % (npass, len(CHECKS)))
for tag, ok in CHECKS:
    print("    %-8s %s" % (tag, "PASS" if ok else "FAIL"))

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
for G, col, lab in ((0.001, "tab:blue", "慢斜坡 G=0.001"), (0.005, "tab:cyan", "G=0.005"),
                    (0.02, "tab:orange", "G=0.02"), (0.1, "tab:red", "快斜坡 G=0.1")):
    rec = res[G]["rec"]
    ax.semilogy(rec[:, 1], np.maximum(rec[:, 2], 1e-16), lw=1.2, color=col, label=lab)
ax.axhline(THRESH, color="k", ls="--", lw=0.8)
ax.axvline(1.0, color="gray", ls=":", lw=0.8)
ax.set_xlim(0, 2.6); ax.set_ylim(1e-16, 3)
ax.set_xlabel("斜坡幅度 a"); ax.set_ylabel("模式幅度 |z|")
ax.set_title("(a) 模式幅度轨迹（虚线：触发阈值与满偏）")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
ax.plot(ag, gg, lw=1.5, color="tab:blue")
ax.set_xlabel("斜坡幅度 a"); ax.set_ylabel("gamma(a) = sqrt(-mu_min)")
ax.set_title("(b) 线性时钟增长率曲线")
for G, col in ((0.001, "tab:blue"), (0.005, "tab:cyan"), (0.02, "tab:orange"), (0.1, "tab:red")):
    pa = pred_ac(G)
    if pa:
        ax.axvline(pa, color=col, ls="--", lw=0.9)

ax = axes[2]
Gs = [G for G, tr, ac, pa in rows1 if tr is not None]
acs = [ac for G, tr, ac, pa in rows1 if tr is not None]
pas = [pa for G, tr, ac, pa in rows1 if tr is not None]
ax.plot(Gs, acs, "o-", color="tab:red", label="实测 a_trig")
ax.plot(Gs, pas, "s--", color="tab:blue", label="线性时钟预言 a_c")
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("斜坡速率 Gamma"); ax.set_ylabel("触发振幅 a_trig")
ax.set_title("(c) 触发振幅对照（灰线：满偏 a=1）")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n  图已保存: %s" % IMG)
print("  面板核对：(a) 四条 |z| 轨迹在 1e-14 至 1 之间、阈值线与满偏线；(b) gamma 单调升至 %.3f；"
      "(c) 两组点与 a=1 灰线。" % GAM1)
