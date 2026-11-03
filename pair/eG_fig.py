"""S0: collect eG_runs/*.json -> eG.json, wall-tension fits, figure eG.png.
Wall-tension approximation: e - e_hom ≈ Γ (σ1+σ2) with σ1,2 = |P| ± |Q| the singular values of G (a relaxed moire is a
network of walls whose length per area scales with the moire wavevectors ~ σ); e_hom = homogeneous elastic energy density of G.
Γ is fitted (least squares through the origin) per (background, K/μ, V0) on all |η/θ|, and also separately on |η/θ| ≤ 0.5
and |η/θ| ≥ 1.5 to show how well a single Γ describes both the normal and the inverted (det G < 0) moire."""
import json,glob,os,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=[json.load(open(f)) for f in sorted(glob.glob('eG_runs/*.json'))]
th=1/30.; mu=1.
def fitG(rows):
    x=np.array([r['sigma1']+r['sigma2'] for r in rows]); y=np.array([r['e']-r['e_hom'] for r in rows])
    return float(np.dot(x,y)/np.dot(x,x)) if len(rows) else None
fits=[]
for a in (90.,0.):
    for lam in (-0.9,-0.5,1.0):
        for V0 in (0.0025,0.005,0.01,0.02):
            S=[r for r in R if r['alpha']==a and r['lam']==lam and r['V0']==V0]
            if not S: continue
            fits.append(dict(alpha=a,lam=lam,K_over_mu=lam+1,V0=V0,n=len(S),Gamma_all=fitG(S),
                             Gamma_small=fitG([r for r in S if r['r']<=0.5]),Gamma_large=fitG([r for r in S if r['r']>=1.5]),
                             Gamma_over_sqrtV0mu=fitG(S)/np.sqrt(V0*mu)))
json.dump(dict(note=__doc__,runs=R,fits=fits),open('eG.json','w'),indent=1)
fig,ax=plt.subplots(2,3,figsize=(17,10))
cols={0.0025:'C0',0.005:'C1',0.01:'C2',0.02:'C3'}; mk={-0.9:'v',-0.5:'o',1.0:'s'}
for i,a in enumerate((90.,0.)):
    for r in R:
        if r['alpha']!=a: continue
        c=cols[r['V0']]; m=mk[r['lam']]; fc=c if r['arg']==0 else 'none'
        ax[i,0].plot(np.sqrt(abs(r['detG']))/th,r['e']/(mu*th**2),m,color=c,mfc=fc,ms=5)
        ax[i,1].plot((r['sigma1']+r['sigma2'])/th,(r['e']-r['e_hom'])/(mu*th**2),m,color=c,mfc=fc,ms=5)
        ax[i,2].plot(r['r'],r['e']/r['e_unrelaxed'] if r['e_unrelaxed']>0 else np.nan,m,color=c,mfc=fc,ms=5)
    for f in fits:
        if f['alpha']!=a: continue
        xs=np.linspace(0,12,10); ax[i,1].plot(xs,f['Gamma_all']*xs*th/(mu*th**2),'-',color=cols[f['V0']],lw=0.6,alpha=0.6)
    bg='twist' if a==90 else 'mismatch'
    ax[i,0].set_xlabel('|det G|^{1/2}/θ'); ax[i,0].set_ylabel('e/(μθ²)'); ax[i,0].set_title(f'{bg}: relaxed energy density')
    ax[i,1].set_xlabel('(σ₁+σ₂)/θ'); ax[i,1].set_ylabel('(e − e_hom)/(μθ²)'); ax[i,1].set_title(f'{bg}: wall-tension form e − e_hom ≈ Γ(σ₁+σ₂) (lines: fits)')
    ax[i,2].set_xlabel('|η/θ|'); ax[i,2].set_ylabel('e / e_unrelaxed'); ax[i,2].set_title(f'{bg}: relaxation gain'); ax[i,2].set_xscale('log')
for V0,c in cols.items(): ax[0,0].plot([],[],'o',color=c,label=f'V0={V0}')
for l,m in mk.items(): ax[0,0].plot([],[],m,color='k',label=f'K/μ={l+1:g}')
ax[0,0].plot([],[],'o',color='k',mfc='none',label='arg η=30° (open)'); ax[0,0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('eG.png',dpi=100)
fm=lambda v: 'nan' if v is None else f'{v:.5f}'
for f in fits: print(f"{'twist' if f['alpha']==90 else 'mismatch':8s} K/μ={f['K_over_mu']:<4g} V0={f['V0']:<6g} n={f['n']} Γ_all={fm(f['Gamma_all'])} Γ(r≤.5)={fm(f['Gamma_small'])} Γ(r≥1.5)={fm(f['Gamma_large'])} Γ/√(μV0)={fm(f['Gamma_over_sqrtV0mu'])}")
