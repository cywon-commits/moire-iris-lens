"""Analyse inclusion runs via exact moire-node positions (u = c1 + lattice vector).
usage: python incl_analyze.py ALPHA LP R0 EPS1 [EPS2 ...]   (needs eps=0 baseline at V0=0.005 and a bare V0=0 run with EPS1>0 at alpha=0)
Output: incl_<alpha>_L<LP>_r<R0>.json and .png
  S_shell(r) = (1/3) * [dU_r(r)/eps] / [U_r^bare(r)/eps_bare]   (bare mismatch has S=K/(K+mu)=1/3 exactly; shapes cancel)"""
import sys,os,json,glob,numpy as np
from nodedisp import node_positions
a=int(sys.argv[1]); Lp=int(sys.argv[2]); r0=int(sys.argv[3]); epss=[float(e) for e in sys.argv[4:]]
def tag(al,e,V):
    rr=[r0] if e!=0 else [r0]+[int(f.split('_r')[1].split('.')[0].split('_')[0]) for f in glob.glob(f'inc_a{al}_e+0.0*_V{V}_L{Lp}_r*.npz')]  # eps=0 baseline is r0-independent
    for fmt,R0 in [(f,x) for x in rr for f in ('+.3f','+.2f')]:
        t=f'inc_a{al}_e{e:{fmt}}_V{V}_L{Lp}_r{R0}'
        if os.path.exists(t+'.npz'): return t
        if os.path.exists(t+'_ck.npz'): print('WARNING unfinished',t); return t+'_ck'
    raise FileNotFoundError(f'inc_a{al}_e{e}_V{V}_L{Lp}_r{r0}')
s3=np.sqrt(3); rm=Lp-12
n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
def ideal(al): z=L/(np.exp(1j*np.radians(al))/30); return z[np.abs(z)<rm]
def Ur(t,al):
    zi=ideal(al); zz,res=node_positions(t,al,rmax=rm); r=np.abs(zi)
    assert np.nanmax(res)<1e-8, f'node finder failed in {t}: {np.nanmax(res)}'
    return r,((zz-zi)*np.conj(zi)/np.maximum(r,1e-9)).real,((zz-zi)*np.conj(zi)/np.maximum(r,1e-9)).imag
def shells(r,v):
    R=np.array(sorted(set(np.round(r[r>1])))); return R,np.array([v[np.abs(r-x)<1].mean() for x in R]),np.array([v[np.abs(r-x)<1].std() for x in R])
bl=glob.glob(f'inc_a0_e+*_V0.0_L{Lp}_r{r0}.npz')
if bl: bare=bl[0][:-4]; eb=float(bare.split('_e')[1].split('_')[0]); rb,ub,_=Ur(bare,0); Rb,Ub,_=shells(rb,ub)
r,u0,t0=Ur(tag(a,0.0,0.005),a); R,U0,_=shells(r,u0)
if not bl: Ub=(1/3)*0.05*r0**2*np.where(R<r0,R/r0**2,1/R)-(1/3)*0.05*r0**2*R/158.**2; eb=0.05   # analytic clamped bare (L_eff~158)
out=dict(alpha=a,Lp=Lp,r0=r0,r=R.tolist(),baseline_Ur_over_r=(U0/R).tolist(),bare_Ur_over_eps=(Ub/eb).tolist(),runs={})
for e in epss:
    t=tag(a,e,0.005); _,ue,te=Ur(t,a); _,dU,dUs=shells(r,ue-u0); _,dT,_=shells(r,te-t0)
    o=(R>r0+8); (Af,Bf),*_=np.linalg.lstsq(np.c_[np.ones(o.sum()),R[o]**2],R[o]*dU[o]/e,rcond=None)   # normalization-free: r dU_r/eps = A + B r^2
    S=(1/3)*(dU/e)/(Ub/eb); E=float(np.load(t+'.npz')['E']) if 'E' in np.load(t+'.npz') else None
    out['runs'][str(e)]=dict(dUr_over_eps=(dU/e).tolist(),spread=(dUs/abs(e)).tolist(),dUt_over_eps=(dT/e).tolist(),S_shell=S.tolist(),
                             S_inside=float(np.mean(S[R<r0-5])),S_outside=float(np.mean(S[R>r0+10])),S_fit=float(Af/r0**2),L_eff=float(np.sqrt(-Af/Bf)) if Af*Bf<0 else None,E=E)
    print(f'eps={e:+.3f}  S_fit=A/r0^2={Af/r0**2:+.3f}  L_eff={np.sqrt(-Af/Bf) if Af*Bf<0 else float("nan"):.0f}  S_inside={np.mean(S[R<r0-5]):+.3f}  S_outside={np.mean(S[R>r0+10]):+.3f}  E={E}')
json.dump(out,open(f'incl_{a}_L{Lp}_r{r0}.json','w'),indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for e in epss: ax[0].plot(R,out['runs'][str(e)]['S_shell'],'o-',label=f'ε*={e:+.3f}')
ax[0].axvline(r0,ls=':',c='gray'); ax[0].set_xlabel('r'); ax[0].set_ylabel('S_shell'); ax[0].legend(); ax[0].set_title(f'α={a}: S per node shell')
es=sorted(epss); ax[1].plot(es,[out['runs'][str(e)]['S_inside'] for e in es],'o-',label='inside'); ax[1].plot(es,[out['runs'][str(e)]['S_outside'] for e in es],'s-',label='outside')
ax[1].set_xlabel('ε*'); ax[1].set_ylabel('S'); ax[1].legend(); ax[1].set_title('amplitude dependence (linear regime = plateau)')
plt.tight_layout(); plt.savefig(f'incl_{a}_L{Lp}_r{r0}.png',dpi=100)
