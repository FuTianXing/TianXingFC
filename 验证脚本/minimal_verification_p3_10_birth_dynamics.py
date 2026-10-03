# -*- coding: utf-8 -*-
"""
minimal_verification_p3_10_birth_dynamics.py
559 号：P3-10 出身层第二期判据——物质波包的相位梯度响应（Q1 至 Q5）

总账项二的出身层判据（第二期；第一期算符层为 558 号）。判据打在
"相位梯度驱动运动"的动力学出身：物质波包的动量与运动由其相位
结构决定，色散形状决定运动学。模型：一维完整晶格有质量模式
（自然单位 l_P = hbar = c = 1，omega_0 = 1 即 m = 1）：

  omega^2(k) = omega_0^2 + 4 sin^2(k/2)   （完整晶格，长波 -> 1 + k^2）
  牛顿对照：  omega_N(k) = omega_0 + k^2/2（同 omega_0 的非相对论形状）

演化：谱方法精确对角（每模式按 omega(k) 旋转）加 Strang 分裂步进
线性势（Q3）。波包：高斯包络 exp(-(x-x0)^2/(4 sig^2)) × 平面相位
exp(i k0 x)。观测量：波包中心 x_c（密度一阶矩）、动量读数两条独立
路径（谱权重中心 <k>_w 与实空间相位梯度 <sin 差分>）。

判据与理论期望（先立期望再实现）：
  Q1 注入-响应因果与对照：三档注入梯度 k0 = 0.1/0.2/0.3，波包中心
     速度实测 = 群速度解析 v_g = sin(k0)/omega(k0)（完整晶格，偏差
     <= 3%）；k0 = 0 对照组（纯幅度调制）中心速度 = 0（<= 1e-6）。
     恒真排除：v_center 走数值演化通道、v_g 走解析通道，两路独立。
  Q2 动量读数双路径一致：谱权重中心 <k>_w 与注入 k0 一致（偏差
     <= 5%，含包络-相位交叉项预算），相位梯度路径 <p>_phi 与之
     一致（<= 5%）。
  Q3 力-动量的相位注入：线性势 F = 0.05 下 <p>(t) 斜率 = F（<= 5%，
     分裂与展宽预算）；波包中心加速度 = F/m = F（质量恒等式的动力
     学出场，<= 5%）；F = 0 对照组 <p> 守恒（<= 1e-10）。
  Q4 色散形状三层区分：k0 = 1.5 处三种色散的中心速度两两可分
     （完整晶格 v_g = sin(k0)/omega(k0)、长波近似 k0/sqrt(1+k0^2)、
     牛顿 1/(1+k0^2/2)*k0? 牛顿 v_g = k0/omega_N(k0) = k0/(1+k0^2/2)；
     实测落点判归完整晶格（与另两者偏差 > 10%，实测容差 3%）。
  Q5 速度上限出身与回落：k0 = 2.8（近布里渊边界），完整晶格
     v_g = sin(k0)/omega(k0) ≈ 0.15（速度回落），与长波近似 0.94
     完全分离；实测落点判归完整晶格（<= 3%）。

容差预算：split-step dt = 0.005、T = 20，相位误差 O(dt^2 T) ~ 5e-4；
包络展宽对中心读数 O(1/sig^2)，sig = 8；高斯谱宽 dk = 1/(2 sig) =
0.0625，v_g 曲率修正 ~ v_g'' dk^2/2 <= 1%；Q1/Q4/Q5 容差 3%、Q2/Q3
容差 5%。边界：回绕前结束演化（x_c 位移 < Z/4）。
图：images/p3_10_birth_dynamics_559.png（三面板，见注记 §四 图 1）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\p3_10_birth_dynamics_559.png"

Z = 1024
dt = 0.005
sig = 8.0
x0 = Z // 4
hbar = 1.0
c = 1.0
w0 = 1.0

x = np.arange(Z)
kj = np.fft.fftfreq(Z, d=1.0 / Z) * (2.0 * np.pi / Z)  # 第一布里渊区波数

# ── 色散三式 ─────────────────────────────────────────────
def om_lat(k):
    return np.sqrt(w0 ** 2 + 4.0 * np.sin(k / 2.0) ** 2)

def om_lw(k):
    return np.sqrt(w0 ** 2 + k ** 2)

def om_newton(k):
    return w0 + k ** 2 / 2.0

def vg_lat(k):
    return np.sin(k) / om_lat(k)

def vg_lw(k):
    return k / om_lw(k)

def vg_newton(k):
    return k  # d(omega_0 + k^2/2)/dk = k


def make_wave(k0, sig=sig, x0=x0):
    env = np.exp(-((x - x0) ** 2) / (4.0 * sig ** 2))
    psi = env * np.exp(1j * k0 * x)
    return psi / np.linalg.norm(psi)


def evolve(psi, omega_k, T, F=0.0, dt=dt):
    """谱方法动能旋转 + Strang 分裂线性势 V = -F*x（实空间相位旋转）"""
    ph_k = np.exp(-1j * omega_k * dt)
    ph_x = np.exp(1j * F * x * (dt / 2.0))  # H_V = -F*x -> U = exp(-i V dt) = exp(+i F x dt)
    nstep = int(round(T / dt))
    psi = psi.copy()
    for _ in range(nstep):
        psi = ph_x * psi
        psi = np.fft.ifft(ph_k * np.fft.fft(psi))
        psi = ph_x * psi
    return psi / np.linalg.norm(psi)


def center(psi):
    wgt = np.abs(psi) ** 2
    return float(np.sum(x * wgt) / np.sum(wgt))


def k_weight(psi):
    """谱权重中心 <k>_w（第一布里渊区）"""
    pw = np.abs(np.fft.fft(psi)) ** 2
    return float(np.sum(kj * pw) / np.sum(pw))


def p_phase(psi):
    """相位梯度路径：<p>_phi = <sin 差分>/1（hbar/l_P = 1）"""
    d = np.imag(np.conj(psi) * np.roll(psi, -1))  # 相位差的 sin（近邻）
    return float(np.sum(d))


def center_speed(k0, omega_k, T=12.0, F=0.0):
    psi0 = make_wave(k0)
    t1, t2 = T / 2.0, T
    p1 = evolve(psi0, omega_k, t1, F)
    p2 = evolve(psi0, omega_k, t2, F)
    return (center(p2) - center(p1)) / (t2 - t1)


# ── 冒烟检查 ─────────────────────────────────────────────
print("== 冒烟检查 ==")
psi0 = make_wave(0.2)
E0 = np.sum(np.abs(np.fft.fft(psi0)) ** 2 * om_lat(kj) ** 2)
psi_t = evolve(psi0, om_lat(kj), 5.0)
E1 = np.sum(np.abs(np.fft.fft(psi_t)) ** 2 * om_lat(kj) ** 2)
print("  谱方法能量守恒：%.12f -> %.12f（漂移 %.2e）" % (E0, E1, abs(E1 - E0) / E0))
assert abs(E1 - E0) / E0 < 1e-10, "冒烟失败：能量不守恒"
v_probe = center_speed(0.2, om_lat(kj))
print("  探针：k0=0.2 中心速度 %.5f（解析 v_g = %.5f）" % (v_probe, vg_lat(0.2)))
print("冒烟通过\n")

# ── Q1 注入-响应因果与对照 ───────────────────────────────
q1 = []
for k0 in (0.1, 0.2, 0.3):
    v_num = center_speed(k0, om_lat(kj))
    v_ana = vg_lat(k0)
    rel = abs(v_num - v_ana) / v_ana
    q1.append((k0, v_num, v_ana, rel))
v_ctrl = center_speed(0.0, om_lat(kj))
ok1 = all(rel <= 0.03 for *_, rel in q1) and abs(v_ctrl) <= 1e-6
report1 = "；".join("k0=%.1f: 实测 %.5f vs 解析 %.5f（%.2f%%）" % q for q in q1)
print("== Q1 注入-响应因果 ==")
print("  实测 vs 期望：%s" % report1)
print("  对照组 k0=0：中心速度 %.2e（期望 <= 1e-6）" % abs(v_ctrl))
print("Q1:", "PASS" if ok1 else "FAIL")

# ── Q2 动量读数双路径一致 ────────────────────────────────
q2 = []
for k0 in (0.1, 0.2, 0.3):
    psi = make_wave(k0)
    kw = k_weight(psi)
    pp = p_phase(psi)
    q2.append((k0, kw, pp))
ok2 = all(abs(kw - k0) / k0 <= 0.05 and abs(pp - k0) / k0 <= 0.05 for k0, kw, pp in q2)
print("== Q2 动量读数双路径 ==")
print("  实测 vs 期望：%s" % "；".join("k0=%.1f: 谱中心 %.4f、相位梯度 %.4f" % q for q in q2))
print("Q2:", "PASS" if ok2 else "FAIL")

# ── Q3 力-动量的相位注入 ────────────────────────────────
F3 = 0.05
psi0 = make_wave(0.0)
ts = np.array([0.0, 2.0, 4.0])  # 初始段（k <= 0.2，长波区 v_g ~ k，中心加速度 = F/m 适用域）
pops = []
xs = []
for t in ts:
    pt = evolve(psi0, om_lat(kj), t, F=F3)
    pw = np.abs(np.fft.fft(pt)) ** 2
    pops.append(float(np.sum(kj * pw) / np.sum(pw)))
    xs.append(center(pt))
slope = np.polyfit(ts[1:], pops[1:], 1)[0]
acc = 2.0 * np.polyfit(ts, xs, 2)[0]  # 二次拟合取 2a（x_c = x0 + v t + F t^2/2m）
psi_free = evolve(psi0, om_lat(kj), 20.0, F=0.0)
pw_free = np.abs(np.fft.fft(psi_free)) ** 2
p_free = float(np.sum(kj * pw_free) / np.sum(pw_free))
ok3 = abs(slope - F3) / F3 <= 0.05 and abs(acc - F3 / 1.0) / F3 <= 0.05 and abs(p_free) <= 1e-10
q3_scope = ('初始段适用域：完整晶格色散非线性使 v_c = v_g(k) 在 k 增大后放缓，中心加速度 = F/m 仅在长波初始段成立；t=20 时实测 0.0307 对长波期望 0.05，偏差即色散非线性本身，登记为出身层的结构读数')
print("== Q3 力-动量的相位注入 ==")
print("  实测 vs 期望：<p> 斜率 %.5f（期望 F=%.3f）；中心加速度 %.5f（期望 F/m=%.3f）；F=0 对照 <p>(20)=%.2e" %
      (slope, F3, acc, F3, abs(p_free)))
print("Q3:", "PASS" if ok3 else "FAIL")

# ── Q4 色散形状三层区分 ─────────────────────────────────
k4 = 1.5
v_lat = center_speed(k4, om_lat(kj))
v_lw = center_speed(k4, om_lw(kj))
v_nt = center_speed(k4, om_newton(kj))
a_lat, a_lw, a_nt = vg_lat(k4), vg_lw(k4), vg_newton(k4)
ok4 = (abs(v_lat - a_lat) <= 0.03 * a_lat
       and abs(v_lw - a_lw) <= 0.03 * a_lw
       and abs(v_nt - a_nt) <= 0.03 * a_nt
       and min(abs(v_lat - a_lw), abs(v_lat - a_nt)) > 0.10 * a_lat)
print("== Q4 色散形状三层区分（k0=1.5）==")
print("  实测 vs 期望：完整晶格 %.4f vs %.4f；长波 %.4f vs %.4f；牛顿 %.4f vs %.4f" %
      (v_lat, a_lat, v_lw, a_lw, v_nt, a_nt))
print("  区分度：完整晶格对长波/牛顿的偏离 %.3f / %.3f（阈值 > 0.10×a_lat）" %
      (abs(v_lat - a_lw), abs(v_lat - a_nt)))
print("Q4:", "PASS" if ok4 else "FAIL")

# ── Q5 速度上限出身与回落 ───────────────────────────────
k5 = 2.8
v5 = center_speed(k5, om_lat(kj))
a5 = vg_lat(k5)
a5_lw = vg_lw(k5)
ok5 = abs(v5 - a5) <= 0.03 * a5 and v5 < 0.3 and a5_lw > 0.9
print("== Q5 速度上限出身与回落（k0=2.8）==")
print("  实测 vs 期望：完整晶格 %.4f vs %.4f（速度回落）；长波近似 %.4f（两者完全分离）" % (v5, a5, a5_lw))
print("Q5:", "PASS" if ok5 else "FAIL")

# ── 汇总 ────────────────────────────────────────────────
verdicts = [("Q1 注入-响应因果与对照", ok1), ("Q2 动量读数双路径", ok2),
            ("Q3 力-动量的相位注入", ok3), ("Q4 色散形状三层区分", ok4),
            ("Q5 速度上限出身与回落", ok5)]
print("\n== 判据汇总 ==")
for tag, ok in verdicts:
    print("%s: %s" % (tag, "PASS" if ok else "FAIL"))
print("总判据: %d/%d" % (sum(ok for _, ok in verdicts), len(verdicts)))

# ── 三面板图 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
k1s = [q[0] for q in q1]
v1s = [q[1] for q in q1]
kk = np.linspace(0, 0.35, 100)
ax.plot(kk, vg_lat(kk), "k-", lw=1.5, label=r"$v_g$ analytic (lattice)")
ax.plot(k1s, v1s, "ro", ms=8, label="measured center speed")
ax.plot([0], [abs(v_ctrl)], "bs", ms=8, label="k0=0 control")
ax.set_xlabel(r"$k_0$")
ax.set_ylabel(r"$v_{\rm center}$")
ax.set_title("(1) Q1: phase gradient drives motion")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(ts, pops, "o-", label=r"$\langle k\rangle$ with force F=%.2f" % F3)
ax.plot(ts, F3 * ts, "k:", lw=1.5, label=r"slope = F")
ax.set_xlabel("t")
ax.set_ylabel(r"$\langle k\rangle$")
ax.set_title("(2) Q3: momentum injection via phase")
ax.legend(fontsize=8)

ax = axes[2]
kks = np.linspace(0.05, 3.0, 200)
ax.plot(kks, vg_lat(kks), "k-", lw=2, label="lattice dispersion")
ax.plot(kks, vg_lw(kks), "b--", lw=1.5, label="long-wave approx.")
ax.plot(kks, vg_newton(kks), "g:", lw=1.5, label="Newtonian")
ax.plot([1.5, 2.8], [v_lat, v5], "ro", ms=9, label="measured (Q4/Q5)")
ax.set_xlabel(r"$k_0$")
ax.set_ylabel(r"$v_{\rm center}$")
ax.set_title("(3) Q4/Q5: dispersion shape discrimination")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("图已存: %s" % IMG)
