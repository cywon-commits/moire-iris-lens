import sys,os,numpy as np
from nodedisp import node_positions
def load(t): return t if os.path.exists(t+'.npz') else t+'_ck'
a=int(sys.argv[1]); e=float(sys.argv[2])
tp=load(f'inc_a{a}_e{e:+.2f}_V0.005_L150_r60'); tm=load(f'inc_a{a}_e{-e:+.2f}_V0.005_L150_r60')
zp,_=node_positions(tp,a,rmax=138); zm,_=node_positions(tm,a,rmax=138); zb,_=node_positions('inc_a0_e+0.05_V0.0_L150_r60',0,rmax=138)
s3=np.sqrt(3); n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
def ideal(al): k=np.exp(1j*np.radians(al))/30; z=L/k; return z[np.abs(z)<138]
zi=ideal(a); zbi=ideal(0)
def radial(z,z0): r=np.abs(z0); return ((z-z0)*np.conj(z0)/np.maximum(r,1e-9)).real, r
Ua,r=radial((zp-zm)/2+zi,zi); Ub,rb=radial(zb,zbi)
out=[]
for R in sorted(set(np.round(r[r>1]))):
    m=np.abs(r-R)<1; mb=np.abs(rb-R)<1
    if mb.sum()==0: continue
    S=(1/3)*(Ua[m].mean()/e)/(Ub[mb].mean()/0.05); out.append((R,S)); print(f'r={R:5.0f} S={S:+.3f}  U_anti/eps={Ua[m].mean()/e:+.2f} spread={np.ptp(Ua[m])/e:.2f}')
