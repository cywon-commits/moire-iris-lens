from style import *
import json
R=[r for r in json.load(open(Dd+'pscan_results.json')) if r.get('status')=='converged' and r.get('rhmode','s')=='s']
fig,axs=plt.subplots(1,2,figsize=(W1,1.75),gridspec_kw=dict(width_ratios=[1.25,1],wspace=0.42,left=0.13,right=0.98,bottom=0.22,top=0.86))
ax=axs[0]; cols={0:'#2e86c1',90:'#e67e22',180:'#c0392b'}
for a,mk in ((0,'o'),(90,'s')):
    for ph in (0,90,180):
        for init in ('a','b'):
            pts=[]
            for c in sorted(set(r['absC'] for r in R)):
                qs=[r['q'] for r in R if r['alpha']==a and r['phim']==ph and r['absC']==c and r['init']==init]
                if not qs: continue
                T=2*np.pi*TH*c/AC; pts.append((T*(1-np.cos(np.radians(ph))),np.mean(qs),np.std(qs)))
            pts=np.array(pts)
            if init=='a':
                ax.errorbar(pts[:,0],pts[:,1],pts[:,2],fmt=mk,ms=3.2,mfc=cols[ph] if a==0 else 'w',mec=cols[ph],ecolor=cols[ph],elinewidth=0.6,capsize=1.2,lw=0)
            else:
                ax.plot(pts[:,0],pts[:,1],'x',ms=3,color='0.55',mew=0.7)
x=np.linspace(0,12,10); ax.plot(x,x,'k--',lw=0.7,zorder=0)
ax.set_xlabel(r'$T(1-\cos\varphi_m)$'); ax.set_ylabel(r'$\langle q\rangle$'); ax.set_xlim(-0.5,12.3); ax.set_ylim(-1.5,17)
for ph,lab in ((0,r'$0^\circ$'),(90,r'$90^\circ$'),(180,r'$180^\circ$')): ax.plot([],[],'o',ms=3.2,color=cols[ph],label=r'$\varphi_m=$'+lab)
ax.plot([],[],'o',ms=3.2,mfc='k',mec='k',label='mismatch'); ax.plot([],[],'s',ms=3.2,mfc='w',mec='k',label='twist'); ax.plot([],[],'x',ms=3,color='0.55',mew=0.7,label='start: background')
ax.legend(loc='upper left',fontsize=5.6,handletextpad=0.2,labelspacing=0.25)
plabel(ax,'(a)','bound integer')
ax=axs[1]; d=np.array([r['q']-r['q_lin'] for r in R if r.get('q_lin') is not None and not (r['place']=='AA' and r['phim']==0)])
ax.hist(d,bins=np.arange(-10.5,3.6,1),color=CG,edgecolor='k',lw=0.4)
ax.set_xlabel(r'$q-q_{\rm lin}$'); ax.set_ylabel('runs'); ax.set_xlim(-10.5,3.5)
ax.text(0.04,0.93,f'{np.mean(d==0)*100:.0f}% exact',transform=ax.transAxes,fontsize=6.5,va='top')
plabel(ax,'(b)','relaxed vs linear')
save(fig,'fig10')
print(len(R),len(d),np.mean(d==0))
