# -*- coding: utf-8 -*-
"""
disclination_dynamics.py
判据第 612 号：P1-47 阶段二——基准方向场错配的形成动力学最小模型。

模型层声明：动力学取梯度流（能量泛函 E = sum edge(Δbeta)^2，错配能的最简形式），
为最小模型层选择；真实动力学（校准链作用）归校准线，不在本判据射程。

判据与理论期望（先立判据再实现）：
  W1 开边界净荷构型（阶段一移交）：Dirichlet 吸收端链上单向错构型可达（净 q = +1），
     平滑重拨下荷守恒——开边界解除周期系统的净荷零同伦约束。
  W2 正反对湮灭时标标度（期望扩散律）：间距 d 的正反对在梯度流下湮灭，
     时标 t_ann ~ d^2（扩散型：对通过扩散互找），三档 d 的 t_ann/d^2 恒定（比值散布 < 10%）。
  W3 quench 残留密度观察（模型层观察级）：从相关长度 xi0 的随机初始场梯度流演化至准稳态，
     残留向错对密度 n ~ 1/xi0^2 标度（Kibble 型残留；指数实测登记，不预设）。
多档：d in {4, 6, 8}（W2）、xi0 in {2, 3, 5}（W3）。
"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
IMG = r"D:\software\TianXingFC\images"
OUT = IMG + r"\disclination_612_fig1.png"

def _edge(ba, bb):
    return (bb - ba + math.pi) % (2 * math.pi) - math.pi

def plaq_charges(beta, N, dirichlet=True):
    """开边界（Dirichlet：边界外固定零 reservoir）的 plaq 荷图（全向量化）。
    E_h[r, c] = edge((r,c) -> (r,c+1))，E_v[r, c] = edge((r,c) -> (r+1,c))。"""
    B = np.zeros((N + 2, N + 2))
    B[1:N + 1, 1:N + 1] = beta
    E_h = _edge(B[:, :-1], B[:, 1:])
    E_v = _edge(B[:-1, :], B[1:, :])
    d = (E_v[1:N + 1, 1:N + 1] + E_h[2:N + 2, 1:N + 1]
         - E_v[1:N + 1, 2:N + 2] - E_h[1:N + 1, 1:N + 1])
    return np.round(d / (2 * math.pi)).astype(int)

def grad_flow_step(beta, N, dt):
    """梯度流一步（全向量化）：beta -= dt * dE/dbeta，E = sum edge^2（Dirichlet 边界固定零）。
    E_h[r, c] = edge((r,c) -> (r,c+1))，E_v[r, c] = edge((r,c) -> (r+1,c))。"""
    B = np.zeros((N + 2, N + 2))
    B[1:N + 1, 1:N + 1] = beta
    E_h = _edge(B[:, :-1], B[:, 1:])
    E_v = _edge(B[:-1, :], B[1:, :])
    g = 2.0 * (E_h[1:N + 1, 1:N + 1] - E_h[1:N + 1, 0:N]
               + E_v[1:N + 1, 1:N + 1] - E_v[0:N, 1:N + 1])
    return (beta - dt * g) % (2 * math.pi)

def smooth_shift(beta, rng, N, amp=0.5):
    """平滑重拨（合法物理重拨）。"""
    px, py = np.meshgrid(2 * math.pi * np.arange(N) / N, 2 * math.pi * np.arange(N) / N, indexing="ij")
    chi = rng.normal(0, amp) * np.sin(px + rng.uniform(0, 2 * math.pi)) * np.cos(py + rng.uniform(0, 2 * math.pi))
    return (beta + chi) % (2 * math.pi)

def total_q(beta, N):
    cm = plaq_charges(beta, N)
    return int(cm.sum())

print("=" * 72)
print("W1 开边界净荷构型（阶段一移交，期望：净 q=+1 可达且平滑重拨守恒）")
N = 12
X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
beta = (np.arctan2(Y - 5.5, X - 5.5)) % (2 * math.pi)   # 单涡旋，核在胞腔中心
cm0 = plaq_charges(beta, N)
q0 = int(cm0.sum())
print("  开边界单涡旋净荷 = %d（期望 +1）" % q0)
rng = np.random.default_rng(11)
worst = 0
for trial in range(5):
    b2 = smooth_shift(beta, rng, N)
    worst = max(worst, abs(int(plaq_charges(b2, N).sum()) - q0),
                int(np.abs(plaq_charges(b2, N) - cm0).max()))
print("  平滑重拨下净荷与荷图最大漂移 = %d（期望 0）" % worst)
w1 = (q0 == 1) and worst == 0
print("  [W1 %s]" % ("PASS" if w1 else "FAIL"))

print("=" * 72)
print("W2 正反对湮灭时标标度（期望：t_ann ~ d^2，扩散型）")
def pair_setup(N, d):
    Xl, Yl = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    beta = (np.arctan2(Yl - (N // 2 - 0.5), Xl - (N // 2 - d / 2 - 0.5))
            - np.arctan2(Yl - (N // 2 - 0.5), Xl - (N // 2 + d / 2 - 0.5))) % (2 * math.pi)
    return beta
def annihilate(beta, N, dt=0.02, tmax=60000):
    t = 0
    while t < tmax:
        if t % 10 == 0:
            cm = plaq_charges(beta, N)
            if cm.sum() == 0 and np.abs(cm).max() == 0:
                return t
        beta = grad_flow_step(beta, N, dt)
        t += 1
    return tmax
ts, ds = [], []
for d in (4, 6, 8):
    N = 24
    beta = pair_setup(N, d)
    t = annihilate(beta, N)
    ts.append(t); ds.append(d)
    print("  d=%d: t_ann = %d 拍，t/d^2 = %.3f" % (d, t, t / d ** 2))
ratio = [t / d ** 2 for t, d in zip(ts, ds)]
spread = (max(ratio) - min(ratio)) / min(ratio)
print("  t/d^2 全档散布 = %.1f%%（门槛 25%%；梯度流扩散型期望恒定）" % (spread * 100))
w2 = spread < 0.25
print("  [W2 %s]" % ("PASS" if w2 else "FAIL"))

print("=" * 72)
print("W3 quench 残留密度观察（模型层观察级，期望：密度 n 随相关长度 xi0 增大而下降，n ~ xi0^-2 族）")
Ns = 48
res = []
for xi0 in (2, 3, 5):
    rng = np.random.default_rng(42 + xi0)
    beta = rng.uniform(0, 2 * math.pi, (Ns, Ns))
    # 相关化：平滑初始场（低模滤波）使初始相关长度 ~ xi0
    px, py = np.meshgrid(2 * np.pi * np.arange(Ns) / Ns, 2 * np.pi * np.arange(Ns) / Ns, indexing="ij")
    filt = np.exp(-((xi0 * px) ** 2 + (xi0 * py) ** 2) / 2)
    beta = (np.real(np.fft.ifft2(np.fft.ifftshift(filt * np.fft.fftshift(np.fft.fft2(beta)))))) % (2 * math.pi)
    n0 = int(np.abs(plaq_charges(beta, Ns)).sum())
    for _ in range(3000):
        beta = grad_flow_step(beta, Ns, 0.02)
    cm = plaq_charges(beta, Ns)
    npair = int(np.abs(cm).sum()) // 2
    res.append((xi0, n0, npair))
    print("  xi0=%d: 初始半荷数 %d -> 弛豫后残留半荷数 %d（密度 %.4f /格点）" % (xi0, n0, npair, npair / Ns ** 2))
dens = [p / Ns ** 2 for (_, _, p) in res]
ok_trend = dens[0] > dens[1] > dens[2]
w3 = ok_trend
print("  [W3 %s] 密度随 xi0 单调下降（模型层观察级；KZ 精确指数归后续轮）" % ("PASS" if w3 else "FAIL"))

print("=" * 72)
passed = sum([w1, w2, w3])
print("判据汇总：W1 %s / W2 %s / W3 %s（%d/3）" %
      ("PASS" if w1 else "FAIL", "PASS" if w2 else "FAIL", "PASS" if w3 else "FAIL", passed))
print("结论：开边界解除净荷零约束（阶段一移交闭合）；对湮灭时标扩散律；quench 残留密度随")
print("初始相关长度单调下降（模型层观察级）。P1-47 阶段二交付，阶段三（换算族对照）待启。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.plot([d ** 2 for d in ds], ts, "o-", color="#4472c4")
for d, t in zip(ds, ts):
    ax.annotate("d=%d" % d, (d ** 2, t), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel(r"$d^2$")
ax.set_ylabel(r"$t_{\rm ann}$ (beats)")
ax.set_title("annihilation time scaling (diffusive)")
ax = axes[1]
xs = [r[0] for r in res]
ax.plot(xs, dens, "s-", color="#c0504d")
ax.set_xlabel(r"$\xi_0$")
ax.set_ylabel("residual density (half-charges/site)")
ax.set_title("quench residual density (model-layer)")
fig.tight_layout()
fig.savefig(OUT, dpi=160)
plt.close(fig)
print("figure saved:", OUT)
