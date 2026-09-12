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
    u = rho - R_param; v = Z
    d = np.sqrt(u**2 + v**2)
    m = np.clip((mask - d)/w, 0.0, 1.0)
    psi = np.arctan2(v, u)
    return mi(n*np.arctan2(m*np.sin(psi), 1.0 + m*(np.cos(psi) - 1.0)))

print("=== 环裁决实验 v2 原始构造 / 原始格点 N=32 / y=0 平面 xz-plaquette ===")
for Rc in (4.0, 3.0, 2.5, 2.0):
    phi = ring_phi(Rc)
    d = pdens(phi[:, JY, :])
    nxs = coords[:d.shape[0]] + 0.5
    half = d[nxs > 0].sum()
    grp = {}
    for u_, v_ in np.argwhere(d != 0):
        x = coords[u_]+0.5; z = coords[v_]+0.5
        grp.setdefault((round(abs(x),1), round(z,1)), []).append(int(d[u_,v_]))
    print(f"  Rc={Rc}: 平面和={d.sum():+.1f}  x>0半平面和={half:+.1f}  非零 plaquette={len(np.argwhere(d!=0))}")
    print("     柱半径分组(z 取 0 邻域):", {k: sum(v) for k, v in sorted(grp.items()) if abs(k[1]) <= 2.0})

N2 = 48
c2 = np.arange(N2) - N2/2.0 + 0.5
X2, Y2, Z2 = np.meshgrid(c2, c2, c2, indexing="ij")
IC = N2//2
RHO2 = np.sqrt(X2**2 + Y2**2)
def ring_masked(Rc, n=1, mask=5.0, w=3.0):
    rp = Rc + mask - w/2.0
    u = RHO2 - rp
    dd = np.sqrt(u**2 + Z2**2)
    m = np.clip((mask - dd)/w, 0.0, 1.0)
    psi = np.arctan2(Z2, u)
    return mi(n*np.arctan2(m*np.sin(psi), 1.0 + m*(np.cos(psi) - 1.0)))
print()
print("=== 我的 N=48 半格偏移 / 平面 y=+0.5 / xz-plaquette ===")
for Rc in (4.0, 3.0):
    d = pdens(np.take(ring_masked(Rc), IC, axis=1))
    nxs = c2[:d.shape[0]] + 0.5
    grp = {}
    for u_, v_ in np.argwhere(d != 0):
        x = c2[u_]+0.5; z = c2[v_]+0.5
        grp.setdefault((round(np.hypot(x,0.5),1), round(z,1)), []).append(int(d[u_,v_]))
    print(f"  Rc={Rc}: 平面和={d.sum():+.1f}  x>0半平面和={d[nxs>0].sum():+.1f}  非零 plaquette={len(np.argwhere(d!=0))}")
    print("     柱半径分组(z 取 0 邻域):", {k: sum(v) for k, v in sorted(grp.items()) if abs(k[1]) <= 2.0})
