from style import *
import json
from matplotlib.patches import Circle, FancyArrowPatch
fig=plt.figure(figsize=(W2,4.3))
ga=fig.add_gridspec(1,2,left=0.02,right=0.985,top=0.945,bottom=0.6,wspace=0.3,width_ratios=[1.15,1])
gb=fig.add_gridspec(1,2,left=0.08,right=0.985,top=0.44,bottom=0.19,wspace=0.3)
gs=[ga[0],ga[1],gb[0],gb[1]]
# (a) schematic
ax=fig.add_subplot(gs[0]); ax.set_aspect('equal'); ax.axis('off')
for cx,typ,col in ((-1.45,'t',CT),(1.45,'r',CM)):
    ax.add_patch(Circle((cx,0),1.0,fc='#f3f3f3',ec='k',lw=0.9))
    for a in np.linspace(0,2*np.pi,12,endpoint=False):
        p=np.array([cx+np.cos(a),np.sin(a)])
        d=np.array([-np.sin(a),np.cos(a)]) if typ=='t' else np.array([np.cos(a),np.sin(a)])
        ax.add_patch(FancyArrowPatch(p,p+0.42*d,arrowstyle='-|>',mutation_scale=6,lw=0.8,color=col))
    ax.plot(cx,0,'k+',ms=4)
ax.text(-1.45,-1.55,'twist ($\\kappa=i\\theta$)\ntangential:\nmicro-torque',ha='center',va='top',fontsize=6.5,color=CT)
ax.text(1.45,-1.55,'mismatch ($\\kappa=\\theta$)\nradial:\nmicro-pressure',ha='center',va='top',fontsize=6.5,color=CM)
ax.text(0.05,1.55,r'$E_1=\oint_{\partial D}(\sigma_{\rm at}n)\cdot\kappa(z-z_0)\,dl$',ha='center',va='bottom',fontsize=7.5)
ax.set_xlim(-3.2,3.2); ax.set_ylim(-2.75,2.05)
plabel(ax,'(a)','ring-traction drive')
# (b) identity vs observed
ax=fig.add_subplot(gs[1])
pts=[('twist',45,-8.82,-8.80),('twist',70,-16.94,-16.84),('mismatch',45,0.877,0.9),('mismatch',60,33.10,32.6),('mismatch',70,7.649,7.76)]
for fam,r0,a,b in pts:
    c=CT if fam=='twist' else CM; mk='s' if fam=='twist' else 'o'
    ax.scatter(a,b,s=22,marker=mk,facecolor=c,edgecolor='k',lw=0.4,zorder=3)
    off={('twist',70):(8,-3),('twist',45):(7,-7),('mismatch',45):(7,-7),('mismatch',70):(7,-7),('mismatch',60):(-7,-3)}[(fam,r0)]
    ax.annotate(f'$r_0={r0}$',(a,b),xytext=off,textcoords='offset points',fontsize=6,ha='left' if off[0]>0 else 'right',va='center')
xx=np.array([-22,38]); ax.plot(xx,xx,'k-',lw=0.5); ax.set_xlim(-22,38); ax.set_ylim(-22,38)
ax.scatter([],[],s=22,marker='s',facecolor=CT,edgecolor='k',lw=0.4,label='twist'); ax.scatter([],[],s=22,marker='o',facecolor=CM,edgecolor='k',lw=0.4,label='mismatch')
ax.legend(loc='upper left'); ax.set_xlabel(r'$E_1$ from ring integral (baseline field)'); ax.set_ylabel(r'$E_1$ from inclusion runs, $\Delta E/2\varepsilon^*$')
plabel(ax,'(b)','identity check')
# (c) placement statistics
ax=fig.add_subplot(gs[2])
Et=json.load(open(Dd+'e1place.json')); Em=json.load(open(Dd+'e1place_a0_L150.json')); Em.update(json.load(open(Dd+'e1place_a0_L200.json')))
rr=np.linspace(0,100,100); ax.plot(rr,2*TH**2*np.pi*rr**2,'-',c=CM,lw=0.9,label=r'$\langle E_1\rangle=2\theta^2\pi r_0^2$')
ax.plot(rr,0*rr,'-',c=CT,lw=0.9,label=r'$\langle E_1\rangle=0$')
for E,c,mk,off in ((Et,CT,'s',-2.2),(Em,CM,'o',2.2)):
    for k,v in E.items():
        r0=float(k); v=np.array(v)
        ax.plot(np.full(len(v)-1,r0+off),v[1:],mk,ms=2.2,mfc='none',mec=c,mew=0.5,alpha=0.8)
        ax.errorbar(r0+off,v.mean(),yerr=v.std()/np.sqrt(len(v)),fmt=mk,ms=4,mfc=c,mec='k',mew=0.4,ecolor='k',elinewidth=0.8,capsize=1.5,zorder=4)
        ax.plot(r0+off,v[0],'*',ms=6,mfc='#f1c40f',mec='k',mew=0.4,zorder=5)
ax.plot([],[],'*',ms=6,mfc='#f1c40f',mec='k',mew=0.4,label='centred ring')
ax.set_xlim(35,103); ax.set_ylim(-25,100); ax.set_xlabel(r'ring radius $r_0$'); ax.set_ylabel(r'$E_1$ over 16 ring centres')
ax.legend(loc='upper left',fontsize=6.0)
plabel(ax,'(c)','placement average')
# (d) S_fit scatter
ax=fig.add_subplot(gs[3])
RF=json.load(open(Dd+'ringfit.json'))
tw=[(float(k.split()[0][3:]),v['S_eff']) for k,v in RF.items()]
I=json.load(open(Dd+'incl_90_L200_r95.json')); tw+=[(95.,v['S_fit']) for v in I['runs'].values()]
mm=[]
for f in ['incl_0_L150_r45','incl_0_L150_r60','incl_0_L150_r70','incl_0_L200_r95']:
    I=json.load(open(Dd+f+'.json')); mm+=[(float(I['r0']),v['S_fit']) for v in I['runs'].values()]
tw=np.array(tw); mm=np.array(mm)
ax.axhspan(-0.149-0.02,-0.149+0.02,color=CT,alpha=0.25,lw=0); ax.axhline(-0.149,c=CT,lw=0.9,label=r'twist $K_M/(K_M+\mu_M)$')
ax.axhline(0.363,c=CM,lw=0.9,ls='--',label=r'mismatch $K_M/(K_M+\mu_M)$')
ax.plot(tw[:,0]-1.5,tw[:,1],'s',ms=3.2,mfc=CT,mec='k',mew=0.3,label='twist runs')
ax.plot(mm[:,0]+1.5,mm[:,1],'o',ms=3.2,mfc=CM,mec='k',mew=0.3,label='mismatch runs')
ax.axvspan(58,62,color=CG,alpha=0.18,lw=0); ax.text(60,8.2,'node\nshell',fontsize=6,color='#555',ha='center',va='top')
ax.set_xlim(38,102); ax.set_ylim(-8.8,8.5); ax.set_xlabel(r'ring radius $r_0$'); ax.set_ylabel(r'far response $S=A/(\varepsilon^* r_0^2)$')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.24),ncol=2,fontsize=6.0)
plabel(ax,'(d)','no universal far response')
save(fig,'fig8')
