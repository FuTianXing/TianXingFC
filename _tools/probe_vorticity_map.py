import numpy as np
N = 48
COORD = np.arange(N, dtype=float) - N / 2.0 + 0.5
IC = N // 2
_X, _Y, _Z = np.meshgrid(COORD, COORD, COORD, indexing="ij")
RHO = np.sqrt(_X**2 + _Y**2)
def mi(d): return (d + np.pi) % (2*np.pi) - np.pi
def pdens(P):
    a00,a10,a11,a01 = P[:-1,:-1],P[1:,:-1],P[1:,1:],P[:-1,1:]
    return (mi(a10-a00)+mi(a11-a10)+mi(a01-a11)+mi(a00-a01))/(2*np.pi)
def ring_naive(Rc): return mi(np.arctan2(_Z, RHO-Rc))
def ring_masked(Rc, mask=5.0, w=3.0, n=1):
    rp = Rc + mask - w/2; u = RHO - rp
    d = np.sqrt(u**2 + _Z**2)
    m = np.clip((mask-d)/w, 0.0, 1.0)
    psi = np.arctan2(_Z, u)
    return mi(n*np.arctan2(m*np.sin(psi), 1.0 + m*(np.cos(psi)-1.0)))

for tag, phi in (("naive", ring_naive(4.0)), ("masked(w=3)", ring_masked(4.0)),
                 ("masked(w=1.5)", ring_masked(4.0, w=1.5))):
    print("="*70); print(tag)
    for axis, s_name in ((1, "y=+0.5"), (0, "x plane")):
        s = IC
        dens = pdens(np.take(phi, s, axis=axis))
        in_ax = [a for a in (0,1,2) if a != axis]
        tot = dens.sum()
        cu = COORD[:dens.shape[0]] + 0.5
        half = dens[(cu > 0)].sum()
        nz = np.argwhere(dens != 0.0)
        print(f"  截面 {s_name}: 归一化平面和 {tot:+.1f}  x>0 半平面和 {half:+.1f}  非零 plaquette {len(nz)}")
        # 近顶点芯位 (4,0) 邻域结构
        pts = []
        for u,v in nz:
            p = [0.0,0.0,0.0]
            p[axis] = COORD[s]; p[in_ax[0]] = COORD[u]+0.5; p[in_ax[1]] = COORD[v]+0.5
            rho = np.hypot(p[0],p[1]); dd = np.hypot(rho-4.0, p[2])
            if dd < 9.0:
                pts.append((round(dd,2), int(dens[u,v]), round(p[in_ax[0]],1), round(p[in_ax[1]],1)))
        pts.sort()
        for dd, val, a, b in pts[:40]:
            print(f"      d_to_core={dd:5.2f}  dens={val:+d}  ({in_ax[0]}={a:5.1f}, {in_ax[1]}={b:5.1f})")
