import os,sys,json,numpy as np
from nodedisp import node_positions
s3=np.sqrt(3); n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); Lv=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
zi=Lv*(-30j); zi=zi[np.abs(zi)<138]; r=np.abs(zi)
def ld(t):
    for x in (t,t+'_ck'):
        if os.path.exists(x+'.npz'): return x
    raise FileNotFoundError(t)
def ur(t): zz,_=node_positions(ld(t),90,rmax=138); return ((zz-zi)*np.conj(zi)/np.maximum(r,1e-9)).real
U0=ur('inc_a90_e+0.00_V0.005_L150_r60'); out={}
for r0,e,tag in [(60,0.05,'inc_a90_e{:+.2f}_V0.005_L150_r60'),(60,0.01,'inc_a90_e{:+.2f}_V0.005_L150_r60'),(70,0.01,'inc_a90_e{:+.3f}_V0.005_L150_r70'),(45,0.01,'inc_a90_e{:+.3f}_V0.005_L150_r45')]:
    for sg in (1,-1):
        try: dU=(ur(tag.format(sg*e))-U0)/(sg*e)
        except FileNotFoundError: continue
        out_=r>r0+8; inn=(r>1)&(r<r0-8)
        M=np.c_[np.ones(out_.sum()),r[out_]**2]; (A,B),*_=np.linalg.lstsq(M,r[out_]*dU[out_],rcond=None)
        c=np.mean(dU[inn]/r[inn]) if inn.any() else np.nan
        key=f'r0={r0} eps={sg*e:+.2f}'; out[key]=dict(S_eff=A/r0**2,Leff=float(np.sqrt(-A/B)) if A*B<0 else None,c_inside=c)
        print(f'{key}:  S_eff=A/(eps r0^2)={A/r0**2:+7.2f}  L_eff={np.sqrt(-A/B) if A*B<0 else float("nan"):6.0f}  inside strain/eps={c:+.2f}')
json.dump(out,open('ringfit.json','w'),indent=1)
