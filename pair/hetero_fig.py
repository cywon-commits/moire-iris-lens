"""P3 figure: heterostrained twist lens (Chang-Refsdal). usage: python hetero_fig.py [CABS R ARG PLACE] -> hetero.png
Rows: phi_m = 0 and 180 deg. Columns: log10 V of the linear field u_b(eta) - C/conj(z-z0) | log10 V of the relaxed run |
source plane beta = (u - c_AA)/kappa = z + (eta/kappa) conj z - m/conj z (m = C/kappa, z measured from z0) with the caustic (astroid) and the
undeformed AA nodes (sources). The critical curve |eta/kappa + m/conj(z)^2| = 1 is drawn on the two maps."""
import sys,os,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex'))
from hexcore import Vhex
from hetero_run import eta_of,htag
from hetero_ana import fields
from pscan_run import place,rh_of,kap,tag
s3=np.sqrt(3.); A=90.
cabs,r,arg,pl=(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]) if len(sys.argv)>4 else (24.,0.3,0.,'AA')
k=kap(A); eta=eta_of(r,arg); g=eta/k; z0=place(pl,A); W=60.
def crit(m,n=2000):
    t=np.linspace(0,2*np.pi,n); zb2=m/(np.exp(1j*t)-g)     # conj(z)^2 on the critical curve
    zb=np.sqrt(zb2); z=np.conj(np.concatenate([zb,-zb])); return z
def beta(z,m): return z+g*np.conj(z)-m/np.conj(z)
fig,ax=plt.subplots(2,3,figsize=(15,9.6))
for i,ph in enumerate((0.,180.)):
    C=cabs*np.exp(1j*np.radians(A+ph)); m=C/k
    f=htag(cabs,ph,r,arg,pl) if r>0 else min([tag(A,cabs,ph,pl,s) for s in 'ab' if os.path.exists(tag(A,cabs,ph,pl,s)+'.npz')],key=lambda t:float(np.load(t+'.npz')['E']))
    D=np.load(f+'.npz'); x=D['x']; sel=np.abs(x)<=W+abs(z0); xs=x[sel]
    _,ul=fields(x,z0,C,rh_of(cabs,'s'),eta)
    for j,(ux,uy,tt) in enumerate(((ul.real,ul.imag,'linear Chang–Refsdal field'),(D['ux'],D['uy'],'relaxed'))):
        V,_,_=Vhex(ux[np.ix_(sel,sel)],uy[np.ix_(sel,sel)],0.005,s3*0.005)
        a=ax[i,j]; im=a.imshow(np.log10(V-V.min()+1e-6).T,origin='lower',extent=[xs[0],xs[-1],xs[0],xs[-1]],cmap='magma',vmin=-4.5,vmax=-1.7)
        zc=crit(m)+z0; a.plot(zc.real,zc.imag,'c.',ms=0.6); a.set_xlim(z0.real-W,z0.real+W); a.set_ylim(z0.imag-W,z0.imag+W)
        a.set_title(f'φ_m={ph:.0f}°: {tt}'); a.set_aspect('equal')
    a=ax[i,2]; zc=crit(m); b=beta(zc,m); a.plot(b.real,b.imag,'r.',ms=0.8,label='caustic')
    n=np.arange(-8,9); N1,N2=np.meshgrid(n,n); Lt=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
    # sources: beta = (u - c_AA)/kappa measured from z0, so undeformed AA nodes sit at L/kappa - z0 - (eta/kappa) conj(z0)
    src=Lt/k-z0-g*np.conj(z0)
    a.plot(src.real,src.imag,'k.',ms=3,label='sources (AA)'); R=np.sqrt(cabs/(1/30.)); L_=2.4*R if ph==180 else 1.5*R; a.set_xlim(-L_,L_); a.set_ylim(-L_,L_); a.set_aspect('equal')
    a.set_title(f'source plane, |η/κ|={r}, arg η={arg:.0f}°'); a.legend(fontsize=8,loc='upper right')
fig.colorbar(im,ax=ax[:,:2],shrink=0.6,label='log10(V − V_min)')
fig.suptitle(f'twist |C|={cabs:g} at {pl}, |η/κ|={r}, arg η={arg:.0f}° (cyan: critical curve)')
plt.savefig('hetero.png',dpi=100,bbox_inches='tight'); print('hetero.png',f)
