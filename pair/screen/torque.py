import sys,numpy as np
from scipy.ndimage import map_coordinates
P='/home/claude/cywon-commits/moire-iris-lens/pair'; sys.path.insert(0,P)
from pscan_run import place
lam,mu=-0.5,1.0
def tau_profile(f,alpha,pl,rs,n=4000):
    D=np.load(f+'.npz'); x=D['x']; h=x[1]-x[0]; ux,uy=D['ux'],D['uy']
    uxx=np.gradient(ux,h,axis=0); uxy=np.gradient(ux,h,axis=1); uyx=np.gradient(uy,h,axis=0); uyy=np.gradient(uy,h,axis=1)
    exx,eyy,exy=uxx,uyy,0.5*(uxy+uyx); tr=exx+eyy
    sxx=lam*tr+2*mu*exx; syy=lam*tr+2*mu*eyy; sxy=2*mu*exy
    z0=place(pl,alpha); out=[]
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    for r in rs:
        X=z0.real+r*np.cos(t); Y=z0.imag+r*np.sin(t); I=(X-x[0])/h; J=(Y-x[0])/h
        g=lambda A: map_coordinates(A,[I,J],order=1)
        nx,ny=np.cos(t),np.sin(t); tx,ty=-ny,nx
        # traction t_i = s_ij n_j ; torque density = r * (traction . t_hat)
        trx=g(sxx)*nx+g(sxy)*ny; try_=g(sxy)*nx+g(syy)*ny
        out.append(np.sum(r*(trx*tx+try_*ty))*(2*np.pi*r/n))
    return np.array(out)
if __name__=='__main__':
    rs=np.arange(4,60,2.0)
    for V0 in ('0.0025','0.005','0.01','0.02'):
        td=tau_profile(f'd_a90_V{V0}_C8_p+0_GEN_L72',90,'GEN',rs); tb=tau_profile(f'bg_a90_V{V0}_L72',90,'GEN',rs)
        tau0=4*np.pi*1.0*8
        print('V0',V0,' '.join('%d:%.2f'%(r,v) for r,v in zip(rs[::2],((td-tb)/tau0)[::2])))
