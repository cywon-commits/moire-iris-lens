from style import *
import json
sys.path.insert(0,'/home/claude/ridge/hex')
from hexcore import Vhex
RP='/home/claude/cywon-commits/moire-iris-lens/pair/'
S=json.load(open(Dd+'screen_scan.json')); L=json.load(open(Dd+'lda_results.json')); EG=json.load(open(Dd+'eG.json'))['runs']
fig=plt.figure(figsize=(W2,4.3))
gt=fig.add_gridspec(1,3,left=0.04,right=0.985,top=0.94,bottom=0.58,wspace=0.32,width_ratios=[1,1,1.25])
gb=fig.add_gridspec(1,2,left=0.07,right=0.985,top=0.45,bottom=0.09,wspace=0.28,width_ratios=[1.5,1])
th=TH; R0=np.sqrt(8/th)
def zgen(): return 0.37*30*np.exp(1j*np.radians(17))
for k,(V0,t) in enumerate(((0.005,'(a)'),(0.02,'(b)'))):
    ax=fig.add_subplot(gt[k]); D=np.load(RP+f'scd_a90_C8_p+0_GEN_V{V0}_l-0.5_a_L150.npz'); x=D['x']
    V,_,_=Vhex(D['ux'],D['uy'],V0,s3*V0); V=V*(0.005/V0); V[D['core']]=np.nan
    z0=zgen(); W=46; m=np.abs(x-z0.real)<W; n=np.abs(x-z0.imag)<W
    ax.imshow(logV(V[np.ix_(m,n)]).T,origin='lower',extent=[x[m][0]-z0.real,x[m][-1]-z0.real,x[n][0]-z0.imag,x[n][-1]-z0.imag],interpolation='bilinear',**LOGV)
    tt=np.linspace(0,2*np.pi,200); ax.plot(R0*np.cos(tt),R0*np.sin(tt),'--',c='#5dade2',lw=0.8)
    rho=[s['rho_node_ext'] for s in S if s['alpha']==90 and abs(s['K_over_mu']-0.5)<1e-6 and s['V0']==V0 and s['init']=='a'][0]
    ax.set_xticks([]); ax.set_yticks([]); plabel(ax,t,r'twist, $V_0=%g$, $\rho=%.2f$'%(V0,rho))
# (c) e(G)
ax=fig.add_subplot(gt[2])
for lam,c,lab in ((-0.9,'#85c1e9',r'$K/\mu=0.1$'),(-0.5,'#2e86c1',r'$0.5$'),(1.0,'#1b4f72',r'$2$')):
    for a,ls,mk in ((90,'-','o'),(0,':','s')):
        pts=sorted({x['r'] for x in EG})
        y=[np.mean([(x['e']-x['e_hom'])/(3*x['V0']) for x in EG if x['alpha']==a and abs(x['lam']-lam)<1e-9 and x['V0']==0.005 and x['r']==r]) for r in pts]
        ax.plot(pts,y,ls,marker=mk,ms=2.5,color=c,label=lab if a==90 else None)
ax.plot([],[],'k-',label='twist'); ax.plot([],[],'k:',label='mismatch')
ax.set_xlabel(r'shear $|\eta|/\theta$'); ax.set_ylabel(r'$(e-e_{\rm hom})/3V_0$'); ax.set_ylim(0.15,1.0)
ax.axvspan(1,5.2,color='0.93',zorder=0); ax.text(3.0,0.2,'inverted',fontsize=6.3,ha='center',color='0.4')
ax.legend(loc='upper left',fontsize=5.8,ncol=2,columnspacing=0.6,handletextpad=0.3)
plabel(ax,'(c)',r'registry energy, sheared moiré')
# (d) rho vs V0
ax=fig.add_subplot(gb[0]); cm={0.1:'#85c1e9',0.5:'#2e86c1',1.0:'#5b2c6f',2.0:'#1b4f72'}
for K in (0.1,0.5,1.0,2.0):
    rows=sorted([s for s in S if s['alpha']==90 and abs(s['K_over_mu']-K)<1e-6 and s['init']=='a'],key=lambda s:s['V0'])
    xs=[max(s['V0'],0.0005)/th**2 for s in rows]; ax.plot(xs,[s['rho_node_ext'] for s in rows],'-o',ms=3,color=cm[K],label=r'twist, $K/\mu=%g$'%K)
for K in (0.1,0.5):
    rows=sorted([s for s in S if s['alpha']==0 and abs(s['K_over_mu']-K)<1e-6 and s['init']=='a'],key=lambda s:s['V0'])
    xs=[max(s['V0'],0.0005)/th**2 for s in rows]; ax.plot(xs,[s['rho_node_ext'] for s in rows],':s',ms=3,mfc='w',color=cm[K],label=r'mismatch, $K/\mu=%g$'%K)
for K in (0.1,0.5,2.0):
    rr=[[l['rho'] for l in L if l['bg']=='twist' and abs(l['K']-K)<1e-6 and l['V0']==V0] for V0 in (0.0025,0.005,0.01,0.02)]
    xs=np.array([0.0025,0.005,0.01,0.02])/th**2
    ax.fill_between(xs,[max(min(r),0) for r in rr],[max(r) for r in rr],color=cm[K],alpha=0.18,lw=0)
ax.set_xscale('log'); ax.set_xlabel(r'registry strength $V_0/\mu\theta^2$  ($V_0=0$ drawn at left)'); ax.set_ylabel(r'$\rho=m_{\rm eff}/m$')
ax.set_ylim(0,1.3); ax.axhline(1,c='k',lw=0.4)
ax.legend(loc='lower left',fontsize=5.8,ncol=2,columnspacing=0.8)
plabel(ax,'(d)',r'screened lens mass ($|C|=8$, $\varphi_m=0$); bands: local-density theory')
# (e) bbar
ax=fig.add_subplot(gb[1])
for l in L:
    if l['rmax']!=0.75: continue
    s=[x for x in S if x['alpha']==(90 if l['bg']=='twist' else 0) and abs(x['K_over_mu']-l['K'])<1e-6 and x['V0']==l['V0'] and x['init']=='a'][0]
    ax.plot(s['b_bar'],l['bbar'],'o' if l['bg']=='twist' else 's',ms=3.2,color=cm[round(l['K'],1)],mfc=cm[round(l['K'],1)] if l['bg']=='twist' else 'w')
x=np.linspace(0,3,5); ax.plot(x,x,'k--',lw=0.6)
ax.set_xlabel(r'measured $\bar b/\theta$'); ax.set_ylabel(r'theory $\bar b/\theta$'); ax.set_xlim(0,2.4); ax.set_ylim(0,3)
plabel(ax,'(e)','inner over-rotation')
save(fig,'fig11')
