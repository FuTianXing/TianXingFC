import numpy as np
def mi(d): return (d + np.pi) % (2*np.pi) - np.pi
def pdens(P):
    a00,a10,a11,a01 = P[:-1,:-1],P[1:,:-1],P[1:,1:],P[:-1,1:]
    return (mi(a10-a00)+mi(a11-a10)+mi(a01-a11)+mi(a00-a01))/(2*np.pi)

N = 32
coords = np.arange(N) - N/2.0
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
JY = N//2
def ring_phi(Rc, n=1, mask=5.0, w=3.0):
    R_param = Rc + mask - w/2.0
    rho = np.sqrt(X**2 + Y**2)
    u = rho - R_param; d = np.sqrt(u**2 + Z**2)
    m = np.clip((mask - d)/w, 0.0, 1.0)
    psi = np.arctan2(Z, u)
    return mi(n*np.arctan2(m*np.sin(psi), 1.0 + m*(np.cos(psi) - 1.0)))

print("=== v2 遮罩构造，N=32 整数格点，y=0 平面（xz-plaquette），阈值 |dens|>0.5 ===")
for Rc in (5.0, 4.0, 3.0, 2.5):
    d = pdens(ring_phi(Rc)[:, JY, :])
    nz = [(coords[u]+0.5, coords[v]+0.5, int(round(d[u,v]))) for u,v in np.argwhere(np.abs(d)>0.5)]
    rho = np.hypot(X[:,JY,:], 0.0)
    print(f"  Rc={Rc}: R_param={Rc+3.5}  非零 plaquette {len(nz)} 条  平面和={d.sum():+.1f}")
    for x, z, val in sorted(nz, key=lambda t: (abs(t[0]), t[1])):
        print(f"      x={x:+6.1f} z={z:+5.1f}  dens={val:+d}")

print()
print("=== 未遮罩 ring_naive，N=32 整数格点，y=0 平面 ===")
def ring_naive(Rc, n=1):
    return mi(n*np.arctan2(Z, np.sqrt(X**2+Y**2) - Rc))
for Rc in (4.0, 3.0):
    d = pdens(ring_naive(Rc)[:, JY, :])
    nz = [(coords[u]+0.5, coords[v]+0.5, int(round(d[u,v]))) for u,v in np.argwhere(np.abs(d)>0.5)]
    print(f"  Rc={Rc}: 非零 plaquette {len(nz)} 条  平面和={d.sum():+.1f}")
    for x, z, val in sorted(nz, key=lambda t: (abs(t[0]), t[1])):
        print(f"      x={x:+6.1f} z={z:+5.1f}  dens={val:+d}")
