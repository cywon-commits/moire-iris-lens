"""S1 figure (tasks/TASK_screening_scan.md sec. 3). usage: python screen_fig.py [screen_scan.json] [screen_scan.png]
(a1,a2) rho_node (extended fit: lens + 1/z dipole + Kelvin force terms) vs V0/(mu theta^2), one line per K/mu,
    init a (filled) and b (open; a and b differ in the fixed boundary values, see S1 results); x: lens-only fit.
(b) rho_node vs theta/(theta+b_bar) with the diagonal rho = theta/(theta+b).
(c) tau(r)/tau0 (solid) and P(r)/tau0 (dotted) for the four representative runs (K/mu=0.5, V0 = 0.005, 0.02)."""
import sys,json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fn=sys.argv[1] if len(sys.argv)>1 else 'screen_scan.json'; out=sys.argv[2] if len(sys.argv)>2 else 'screen_scan.png'
R=[r for r in json.load(open(fn)) if r.get('status')=='converged' and r.get('node')]
th=1/30.; D={(r['alpha'],r['lam'],r['V0'],r['init']):r for r in R}
cols={-0.9:'C0',-0.5:'C1',0.0:'C2',1.0:'C3'}; bgname={90.0:'twist',0.0:'mismatch'}
fig,ax=plt.subplots(2,2,figsize=(13,10)); ax=ax.ravel()
for i,al in enumerate((90.0,0.0)):
    for lam,c in cols.items():
        for init,ls,mf in (('a','-','full'),('b',':','none')):
            ks=sorted(k for k in D if k[0]==al and k[1]==lam and k[3]==init)
            if not ks: continue
            x=np.array([k[2]/th**2 for k in ks]); xp=np.where(x>0,x,0.5); y=[D[k]['rho_node_ext'] for k in ks]
            ax[i].plot(xp,y,ls,color=c,marker='o',mfc=c if mf=='full' else 'none',label=f'K/μ={lam+1:g} init {init}')
            if init=='a': ax[i].plot(xp,[D[k]['rho_node'] for k in ks],ls='none',marker='x',color=c,ms=5)
            b=[1/(1+D[k]['b_bar']) for k in ks]; ax[2].plot(b,y,'o' if al==90 else 's',color=c,mfc=c if init=='a' else 'none',alpha=0.8)
    ax[i].set_xscale('log'); ax[i].set_xlabel('V0/(μθ²)  (V0=0 drawn at 0.5)'); ax[i].set_ylabel('ρ_node (extended fit)'); ax[i].set_ylim(0,1.5); ax[i].legend(fontsize=7,ncol=2)
    ax[i].set_title(f'(a{i+1}) {bgname[al]}: o-o ext. fit (init a filled, b open), x lens-only fit (a)')
ax[2].plot([0,1.05],[0,1.05],'k:'); ax[2].set_xlabel('θ/(θ+b̄)'); ax[2].set_ylabel('ρ_node (extended fit)'); ax[2].set_xlim(0.25,1.02); ax[2].set_ylim(0,1.5)
ax[2].set_title('(b) circles twist, squares mismatch; filled init a, open init b')
for al,V,c in ((90.0,0.005,'C0'),(90.0,0.02,'C1'),(0.0,0.005,'C2'),(0.0,0.02,'C3')):
    r=D.get((al,-0.5,V,'a'))
    if not r: continue
    T=r['torque']; rr=np.array(T['r'])/r['R']; ax[3].plot(rr,T['tau'],'-',color=c,label=f"{bgname[al]} V0={V:g}"); ax[3].plot(rr,T['P'],':',color=c)
ax[3].axhline(0,color='k',lw=0.5); ax[3].set_xlabel('r/R'); ax[3].set_ylabel('τ/τ0 (solid), P/τ0 (dotted)'); ax[3].legend(fontsize=8); ax[3].set_title('(c) K/μ=0.5, init a, τ0=4πμ|C|')
fig.tight_layout(); fig.savefig(out,dpi=110); print('wrote',out)
