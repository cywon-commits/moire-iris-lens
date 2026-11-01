from style import *
import json
fit=json.load(open(Dd+'permod2_fit.json'))
fig=plt.figure(figsize=(W2,2.35))
gs=fig.add_gridspec(1,4,left=0.06,right=0.985,top=0.88,bottom=0.33,wspace=0.45)
cols={'d':'#1f77b4','w':'#d62728','g1':'#2ca02c','g2':'#9467bd'}
lab={'d':r'dilation ${\rm Re}\,U_a$ (minus linear term)','w':r'rotation ${\rm Im}\,U_a$','g1':r'shear ${\rm Re}\,U_b$','g2':r'shear ${\rm Im}\,U_b$'}
for k,(a,t) in enumerate(((90,'twist'),(0,'mismatch'))):
    ax=fig.add_subplot(gs[k]); D=json.load(open(Dd+f'permod2_a{a}.json')); e0=[x['e'] for x in D if x['Ua']==[0,0] and x['Ub']==[0,0]][0]; F=fit[str(a)]
    for mo in cols:
        pts=[]
        for x in D:
            v={'d':x['Ua'][0],'w':x['Ua'][1],'g1':x['Ub'][0],'g2':x['Ub'][1]}
            if all(abs(v[q])<1e-12 for q in v if q!=mo): pts.append((v[mo],x['e']))
        pts=np.array(sorted(pts)); lin=F['sig_d']*TH**2*pts[:,0] if mo=='d' else 0
        ax.plot(pts[:,0]*100,(pts[:,1]-e0-lin)/TH**2*1e3,'o-',c=cols[mo],ms=2.6,lw=0.9,label=lab[mo])
    ax.axhline(0,c='k',lw=0.4); ax.set_xlabel(r'strain amplitude ($10^{-2}$)'); ax.set_ylabel(r'$(e-e_0)/\theta^2$ ($10^{-3}$)')
    ax.text(0.5,0.97,f"$K_M={F['K']:+.2f}$, $\\mu_M={F['mu']:.2f}$\n$\\chi_M={F['chi']:+.2f}$ ($\\theta^2$)",transform=ax.transAxes,ha='center',va='top',fontsize=6.2,linespacing=1.2)
    plabel(ax,'(a)' if k==0 else '(b)',t+' moiré')
    if k==0: ax.set_ylim(-0.6,4.2)
    else: ax.set_ylim(-0.8,1.5)
H,Lb=ax.get_legend_handles_labels()
fig.legend(H,Lb,loc='lower center',bbox_to_anchor=(0.27,0.0),ncol=2,fontsize=6.2,columnspacing=1.0)
# (c) bulk expansion vs 1/L
ax=fig.add_subplot(gs[2])
tw=[(72,0.0386),(148.5,0.018),(150,0.0198),(200,0.0166)]; mm=[(72,0.033),(148.5,0.013),(150,0.0081),(200,0.0091)]
for dat,c,mk,l in ((tw,CT,'s','twist'),(mm,CM,'o','mismatch')):
    d=np.array(dat); ax.plot(1000/d[:,0],d[:,1]*100,mk,c=c,mfc=c,mec='k',mew=0.3,ms=4,label=l)
xx=np.linspace(0,15,10); ax.plot(xx,2.75*xx/10,'k--',lw=0.7,label=r'$\Delta^*/L$, $\Delta^*=2.75$')
ax.set_xlim(0,15); ax.set_ylim(0,4.6); ax.set_xlabel(r'$10^3/L$ (box half-width $L$)'); ax.set_ylabel(r'bulk dilation $\varepsilon_0$ ($10^{-2}$)')
ax.legend(loc='upper left',fontsize=6.0)
plabel(ax,'(c)','defect-free box')
# (d) boundary dilation following
ax=fig.add_subplot(gs[3]); E=json.load(open(Dd+'eta_scan.json'))
for key,c,mk,l in (('90_54','#e59866','^','twist, $L=54$'),('90_72',CT,'s','twist, $L=72$')):
    d=np.array([(x['eta'],x['eps_b']) for x in E[key]]); ax.plot(d[:,0]*100,d[:,1]*100,mk+'-',c=c,ms=3.2,mec='k',mew=0.3,lw=0.8,label=l)
xx=np.linspace(-3,3,5); ax.plot(xx,xx+3.86,'k--',lw=0.7,label='slope 1 (stiff edge)')
ax.set_xlabel(r'boundary dilation $\eta$ ($10^{-2}$)'); ax.set_ylabel(r'bulk dilation $\varepsilon_b$ ($10^{-2}$)')
ax.set_xlim(-3.3,3.3); ax.set_ylim(0,15); ax.legend(loc='upper left',fontsize=6.0)
plabel(ax,'(d)','bulk follows clamp')
save(fig,'fig7')
