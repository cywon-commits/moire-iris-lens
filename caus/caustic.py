import numpy as np
from scipy.optimize import fsolve
def lensmap(z,a,m): return z+a*np.conj(z)-m/np.conj(z)
def detJ(z,a,m): return 1-np.abs(a+m/np.conj(z)**2)**2
def critical(a,m,n=4000):
    ph=np.linspace(0,2*np.pi,n,endpoint=False); e=np.exp(2j*ph)
    # |a + m e^{2i phi} s|^2 = 1 with s = 1/r^2 (z = r e^{i phi}, 1/conj(z)^2 = e^{2i phi}/r^2)
    A=abs(m)**2; B=2*np.real(np.conj(a)*m*e); Cc=abs(a)**2-1
    disc=B*B-4*A*Cc; s=(-B+np.sqrt(np.maximum(disc,0)))/(2*A)
    z=np.exp(1j*ph)/np.sqrt(s); return z, lensmap(z,a,m)
def count_map(a,m,L=3.0,nb=240,zmax=8.0,nz=1400,rmin=0.05):
    """image count n(beta) on a beta grid via area-weighted forward mapping (images with |z|>rmin)"""
    x=np.linspace(-zmax,zmax,nz); X,Y=np.meshgrid(x,x,indexing='ij'); Z=X+1j*Y; dA=(x[1]-x[0])**2
    ok=np.abs(Z)>rmin; Z=Z[ok]; B=lensmap(Z,a,m); J=np.abs(detJ(Z,a,m))
    sign=np.sign(detJ(Z,a,m))
    edges=np.linspace(-L,L,nb+1); db=(edges[1]-edges[0])**2
    H,_,_=np.histogram2d(B.real,B.imag,bins=[edges,edges],weights=J*dA/db)
    Hp,_,_=np.histogram2d(B.real,B.imag,bins=[edges,edges],weights=(J*dA/db)*(sign>0))
    return edges,H,Hp,H-Hp
def solve_images(beta,a,m,R=1.0):
    sols=[]
    for r0 in np.linspace(0.15,4,14)*R:
        for p0 in np.linspace(0,2*np.pi,16,endpoint=False):
            def F(v):
                z=v[0]+1j*v[1]; w=lensmap(z,a,m)-beta; return [w.real,w.imag]
            v,info,ier,msg=fsolve(F,[r0*np.cos(p0),r0*np.sin(p0)],full_output=True)
            if ier==1:
                z=v[0]+1j*v[1]
                if abs(z)>1e-3 and abs(lensmap(z,a,m)-beta)<1e-9 and all(abs(z-s)>1e-5 for s in sols): sols.append(z)
    return sols
if __name__=="__main__":
    R=1.0
    for name,phim in [('converging',0.),('twist (vacancy)',-90.),('diverging',180.)]:
        m=R**2*np.exp(1j*np.radians(phim))
        for a in [0.2,0.5,0.2j,0.5*np.exp(0.25j*np.pi)]:
            zc,bc=critical(a,m)
            # cusps: where caustic tangent reverses -> local extrema of arg-speed; count sign changes of d|dbeta|
            db=np.gradient(bc); sp=np.abs(db); cusp=int(np.sum((sp<np.roll(sp,1))&(sp<np.roll(sp,-1))&(sp<0.05*sp.max())))
            edges,H,Hp,Hm=count_map(a,m)
            vals=np.round(H[H>0.5]); u,cn=np.unique(vals,return_counts=True)
            print(f'{name:16s} a={a:.2f}  cusps~{cusp}  image counts present: '+', '.join(f'{int(v)}' for v,c in zip(u,cn) if c>20)+
                  f'   (n+ - n-) values: {sorted(set(np.round((Hp-Hm)[H>0.5]).astype(int).tolist()))}')
