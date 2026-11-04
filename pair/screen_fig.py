"""S1 figure (tasks/TASK_screening_scan.md sec. 3). usage: python screen_fig.py [screen_scan.json] [screen_scan.png]
(a) rho_node vs V0/(mu theta^2), one line per K/mu, twist (solid) and mismatch (dashed); lower-energy init per point,
    open symbols: extended fit (with 1/z dipole and Kelvin force terms).
(b) rho_node vs theta/(theta+b_bar) with the diagonal rho = theta/(theta+b).
(c) tau(r)/tau0 (solid) and P(r)/tau0 (dotted) for the four representative runs (K/mu=0.5, V0 = 0.005, 0.02)."""
import sys,json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fn=sys.argv[1] if len(sys.argv)>1 else 'screen_scan.json'; out=sys.argv[2] if len(sys.argv)>2 else 'screen_scan.png'
R=[r for r in json.load(open(fn)) if r.get('status')=='converged' and r.get('node')]
th=1/30.; best={}
for r in R:
    k=(r['alpha'],r['lam'],r['V0'])
    if k not in best or r['dE']<best[k]['dE']: best[k]=r
cols={-0.9:'C0',-0.5:'C1',0.0:'C2',1.0:'C3'}; bgname={90.0:'twist',0.0:'mismatch'}
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
for (al,ls) in ((90.0,'-'),(0.0,'--')):
    for lam,c in cols.items():
        ks=sorted(k for k in best if k[0]==al and k[1]==lam)
        if not ks: continue
        x=np.array([k[2]/th**2 for k in ks]); y=[best[k]['rho_node'] for k in ks]; ye=[best[k]['rho_node_ext'] for k in ks]
        xp=np.where(x>0,x,0.5); ax[0].plot(xp,y,ls,color=c,marker='o',label=f'{bgname[al]} K/μ={lam+1:g}')
        ax[0].plot(xp,ye,ls='none',color=c,marker='o',mfc='none',ms=7)
        b=[1/(1+best[k]['b_bar']) for k in ks]; ax[1].plot(b,y,'o' if al==90 else 's',color=c,mfc=c if al==90 else 'none')
ax[0].set_xscale('log'); ax[0].set_xlabel('V0/(μθ²)  (V0=0 drawn at 0.5)'); ax[0].set_ylabel('ρ_node'); ax[0].set_ylim(0,1.5); ax[0].legend(fontsize=7,ncol=2)
ax[0].set_title('(a) ρ_node (filled/lines), extended fit (open)')
ax[1].plot([0,1.05],[0,1.05],'k:'); ax[1].set_xlabel('θ/(θ+b̄)'); ax[1].set_ylabel('ρ_node'); ax[1].set_xlim(0.25,1.02); ax[1].set_ylim(0,1.5)
ax[1].set_title('(b) circles: twist, squares: mismatch')
for al,V,c in ((90.0,0.005,'C0'),(90.0,0.02,'C1'),(0.0,0.005,'C2'),(0.0,0.02,'C3')):
    r=best.get((al,-0.5,V))
    if not r: continue
    T=r['torque']; rr=np.array(T['r'])/r['R']; ax[2].plot(rr,T['tau'],'-',color=c,label=f"{bgname[al]} V0={V:g}"); ax[2].plot(rr,T['P'],':',color=c)
ax[2].axhline(0,color='k',lw=0.5); ax[2].set_xlabel('r/R'); ax[2].set_ylabel('τ/τ0 (solid), P/τ0 (dotted)'); ax[2].legend(fontsize=8); ax[2].set_title('(c) K/μ=0.5, τ0=4πμ|C|')
fig.tight_layout(); fig.savefig(out,dpi=130); print('wrote',out)
