import numpy as np
def mi(d): return (d + np.pi) % (2*np.pi) - np.pi
def pdens(P):
    a00,a10,a11,a01 = P[:-1,:-1],P[1:,:-1],P[1:,1:],P[:-1,1:]
    return (mi(a10-a00)+mi(a11-a10)+mi(a01-a11)+mi(a00-a01))/(2*np.pi)

# ---- (3) 整数格点的 ±π 伪影 ----
N = 32
co = np.arange(N) - N/2.0
X, Y, Z = np.meshgrid(co, co, co, indexing="ij")
RHO = np.sqrt(X**2 + Y**2)
phi = mi(np.arctan2(Z, RHO - 4.0))
for jl, lab in ((16, "y=0 平面(整数格面)"),):
    d = pdens(phi[:, jl, :])
    vals = [(co[u]+0.5, co[v]+0.5, d[u,v]) for u,v in np.argwhere(np.abs(d) > 1e-6)]
    print(f"[整数格点 N=32 未遮罩环 Rc=4] {lab}: 非整数/非零 plaquette {len(vals)} 条，y=0 处相位恰为 ±π 的边会导致 mi 失对称")
    for x, z, v in sorted(vals, key=lambda t: -abs(t[2]))[:12]:
        print(f"      x={x:+6.1f} z={z:+5.1f}  dens={v:+.6f}")
    print("      平面和 =", f"{d.sum():+.6f}")

# ---- (1)(2) 半格偏移格点上两种构造的真实涡量 ----
N2 = 48
c2 = np.arange(N2) - N2/2.0 + 0.5
X2, Y2, Z2 = np.meshgrid(c2, c2, c2, indexing="ij")
IC = N2//2
RHO2 = np.sqrt(X2**2 + Y2**2)
def naive(Rc, n=1): return mi(n*np.arctan2(Z2, RHO2 - Rc))
def masked(Rc, n=1, mask=5.0, w=3.0):
    rp = Rc + mask - w/2.0
    u = RHO2 - rp; dd = np.sqrt(u**2 + Z2**2)
    m = np.clip((mask - dd)/w, 0.0, 1.0)
    psi = np.arctan2(Z2, u)
    return mi(n*np.arctan2(m*np.sin(psi), 1.0 + m*(np.cos(psi) - 1.0)))
def E(phi):
    e = 0.6*float(np.sum(1.0-np.cos(phi)))
    for ax in range(3):
        e += 0.5*float(np.sum(1.0-np.cos(mi(np.diff(phi, axis=ax)))))
    return e

for tag, phi in (("未遮罩", naive(4.0)), ("遮罩 v2", masked(4.0))):
    d = pdens(np.take(phi, IC, axis=1))
    real = [(np.hypot(c2[u]+0.5, 0.5), c2[v]+0.5, d[u,v]) for u,v in np.argwhere(np.abs(d) > 0.5)]
    print()
    print(f"[半格偏移 N=48 平面 y=+0.5] {tag} 环 Rc=4: 真实涡量 plaquette {len(real)} 条; E0={E(phi):.1f}")
    for rho, z, v in sorted(real, key=lambda t: t[0]):
        print(f"      rho={rho:5.2f} z={z:+5.1f}  dens={v:+.1f}")
    cu = c2[:d.shape[0]] + 0.5
    print(f"      平面和={d.sum():+.1f}  x>0 半平面和={d[cu>0].sum():+.1f}")
