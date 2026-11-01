from style import *
import json
C=json.load(open(B+'ceff_compare.json')); K=json.load(open(B+'k_results.json')); R=json.load(open(B+'big_res.json'))
fig=plt.figure(figsize=(W2,4.2))
ga=fig.add_gridspec(1,2,left=0.08,right=0.985,top=0.945,bottom=0.655,wspace=0.27); gb=fig.add_gridspec(1,2,left=0.08,right=0.985,top=0.43,bottom=0.1,wspace=0.27)
axs=np.array([[fig.add_subplot(ga[0]),fig.add_subplot(ga[1])],[fig.add_subplot(gb[0]),fig.add_subplot(gb[1])]])
sm=lambda y,n=61: np.convolve(y,np.ones(n)/n,'same')
col={'pull':'#d35400','push':'#1e8449'}
for ax,fam,key,tt in ((axs[0,0],'tw','d40','(a) twist pair $d=40$: dressed charge'),(axs[0,1],'mm','d26','(b) mismatch pair $d=26$: dressed charge')):
    for br in ('pull','push'):
        n=f'{fam} {br} {key}'; r=np.array(C[n]['r']); k=(r>=22)&(r<=138)
        ax.plot(r[k],sm(C[n]['ReC'])[k],'-',c=col[br],lw=1.3)
        ax.plot(r[k],sm(C[n]['ImC'])[k],'--',c=col[br],lw=1.3)
    cb=K[0]['Cbare'] if fam=='tw' else K[2]['Cbare']
    rr=np.linspace(22,138,50); Lfit=160.
    ax.plot(rr,cb*(1-rr**2/Lfit**2),':',c='k',lw=1.0)
    ax.axhline(0,c='k',lw=0.4); ax.set_xlim(20,140); ax.set_xlabel(r'loop radius $\rho$ (box half-width 148.5)'); ax.set_ylabel(r'$C_{\rm eff}(\rho)$')
    plabel(ax,tt[:3],tt[4:])
axs[0,0].set_ylim(-6,12.5); axs[0,1].set_ylim(-6,12.5)
from matplotlib.lines import Line2D
H=[Line2D([],[],c=col['pull'],lw=1.3),Line2D([],[],c=col['push'],lw=1.3),Line2D([],[],c='k',lw=1.3),Line2D([],[],c='k',lw=1.3,ls='--'),Line2D([],[],c='k',lw=1.0,ls=':')]
fig.legend(H,['pulled branch','pushed branch','Re $C_{\\rm eff}$ (area)','Im $C_{\\rm eff}$ (torque)',r'bare $\sum C$ in clamped box'],loc='center',bbox_to_anchor=(0.53,0.525),ncol=5,fontsize=6.5,columnspacing=1.2)
# (c) force-dipole formula
ax=axs[1,0]
for k in K:
    tw=k['alpha']==90; k2='K/mu=2' in k['name']
    if (not tw) and k2: continue
    mk='o' if not k2 else 'D'; c=CT if tw else CM
    ax.scatter(k['Creg_dip'][0],k['Creg_meas'][0],marker=mk,s=22,facecolor=c,edgecolor='k',lw=0.4,zorder=3)
    if tw: ax.scatter(k['Creg_dip'][1],k['Creg_meas'][1],marker=mk,s=22,facecolor='w',edgecolor=c,lw=1.0,zorder=3)
xx=np.array([-5,8.5]); ax.plot(xx,xx,'k-',lw=0.5)
for lab,mk,fc,ec in (('twist, Re (area)','o',CT,'k'),('twist, Im (torque)','o','w',CT),('mismatch, Re','o',CM,'k'),(r'$K/\mu=2$ (diamonds)','D','#bbbbbb','k')):
    ax.scatter([],[],marker=mk,s=22,facecolor=fc,edgecolor=ec,lw=0.6,label=lab)
ax.set_xlim(-5,8.5); ax.set_ylim(-5,8.5); ax.legend(loc='upper left',fontsize=6.0)
ax.set_xlabel(r'force-dipole formula $-\frac{{\rm Tr}P}{4\pi(\lambda+2\mu)}-i\frac{M}{4\pi\mu}$'); ax.set_ylabel(r'measured $C_{\rm reg}=C_0-C_{\rm bare}$')
plabel(ax,'(c)','induced charge from registry forces')
# (d) K independence
ax=axs[1,1]
lab=['pulled','pushed']; x=np.arange(2)
re05=[K[0]['Creg_meas'][0],K[1]['Creg_meas'][0]]; re2=[K[4]['Creg_meas'][0],K[5]['Creg_meas'][0]]
im05=[K[0]['Creg_meas'][1],K[1]['Creg_meas'][1]]; im2=[K[4]['Creg_meas'][1],K[5]['Creg_meas'][1]]
w=0.18
ax.bar(x-1.5*w,re05,w,color='#999999',label=r'Re, $K/\mu=0.5$'); ax.bar(x-0.5*w,re2,w,color='#444444',label=r'Re, $K/\mu=2$')
ax.bar(x+0.5*w,im05,w,color='#d2b4de',label=r'Im, $K/\mu=0.5$'); ax.bar(x+1.5*w,im2,w,color='#7d3c98',label=r'Im, $K/\mu=2$')
ax.scatter(x-0.5*w,np.array(re05)*0.5,marker='_',s=120,c='r',lw=1.5,zorder=4,label=r'Re if force-controlled ($\propto 1/(\lambda+2\mu)$)')
ax.axhline(0,c='k',lw=0.5); ax.set_xticks(x); ax.set_xticklabels(['pulled branch','pushed branch'])
ax.set_ylabel(r'measured $C_{\rm reg}$ (twist, $d=40$)'); ax.set_ylim(-6.5,15.5); ax.set_xlim(-0.5,1.5)
ax.legend(loc='upper right',fontsize=5.9,ncol=2,columnspacing=0.6)
plabel(ax,'(d)','stiffness independence of $C_{\\rm reg}$')
save(fig,'fig5')
