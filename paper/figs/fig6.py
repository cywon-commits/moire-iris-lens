from style import *
import json
S=json.load(open(P+'vac_count.json'))
fig=plt.figure(figsize=(W2,2.45))
gs=fig.add_gridspec(1,3,left=0.07,right=0.99,top=0.89,bottom=0.17,wspace=0.36,width_ratios=[1,1,0.95])
ax=fig.add_subplot(gs[0])
for s in S:
    c=CT if s['fam']=='twist' else CM; far=s['d']>40
    ax.scatter(s['F_count'],s['F'],marker='x' if far else 'o',s=14,facecolor=c if not far else c,edgecolor='k' if not far else None,lw=0.3 if not far else 0.8,zorder=3)
xx=np.array([-3.4,1.4]); ax.plot(xx,xx,'k-',lw=0.5); ax.set_xlim(-3.4,1.4); ax.set_ylim(-3.4,1.4)
ax.scatter([],[],s=14,facecolor=CT,edgecolor='k',lw=0.3,label='twist'); ax.scatter([],[],s=14,facecolor=CM,edgecolor='k',lw=0.3,label='mismatch')
ax.scatter([],[],marker='x',s=14,c='k',lw=0.8,label='$d\\geq 46$')
ax.set_xlabel(r'windowed count $\sum Q_{\rm hole}-N_{-}-\Delta N_{+}$'); ax.set_ylabel(r'$F=-\frac{2\pi}{A_c}{\rm Re}(\bar\kappa C_{\rm eff})$')
ax.legend(loc='lower right',fontsize=6.0)
plabel(ax,'(a)','far charge = node bookkeeping')
ax=fig.add_subplot(gs[1]); rng=np.random.default_rng(3)
T=[s for s in S if s['fam']=='twist' and 16<=s['d']<=46]
for s in T:
    mz=min(s['z']); c='#1e8449' if mz<=6 else '#8e44ad'
    ax.scatter(s['sumq']+rng.uniform(-.1,.1),s['F'],color=c,s=16,edgecolor='k',lw=0.3,zorder=3)
ax.scatter([],[],color='#1e8449',s=16,edgecolor='k',lw=0.3,label='a core with $z\\leq 6$'); ax.scatter([],[],color='#8e44ad',s=16,edgecolor='k',lw=0.3,label='both cores $z\\geq 8$')
ax.axhline(0,c='k',lw=0.5); ax.set_xticks([1,2,3]); ax.set_xlim(0.5,3.5); ax.set_ylim(-2.0,2.4)
ax.set_xlabel(r'bound nodes $\sum q$'); ax.set_ylabel('$F$'); ax.legend(loc='upper left',fontsize=6.0)
plabel(ax,'(b)','sign rule, twist pairs')
ax=fig.add_subplot(gs[2]); cwd=os.getcwd(); os.chdir(P)
from burgers import circuits
D=np.load('swout_d40.0.npz'); x=D['x']; m=np.abs(x)<58; V=Vmap(D)
ax.imshow(logV(V[np.ix_(m,m)]).T,origin='lower',extent=[x[m][0],x[m][-1]]*2,interpolation='bilinear',**LOGV)
res,pos,lab=circuits('swout_d40.0',[(-20.,0.),(20.,0.)],90)
os.chdir(cwd)
for c,Bv in res:
    col='#00e5ff' if (Bv[0]+0.5*Bv[1])>0 or (Bv[0]==0 and Bv[1]>0) else '#7dff4d'
    ax.plot(c.real,c.imag,'h',ms=8,mfc='none',mec=col,mew=1.3)
ax.plot([],[],'h',ms=7,mfc='none',mec='#00e5ff',mew=1.2,label=r'$+\mathbf{B}$'); ax.plot([],[],'h',ms=7,mfc='none',mec='#7dff4d',mew=1.2,label=r'$-\mathbf{B}$')
ax.set_xlim(-58,58); ax.set_ylim(-58,58); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.01),ncol=2,fontsize=6.3,handletextpad=0.2)
plabel(ax,'(c)','moiré Burgers dipole')
save(fig,'fig6'); print(len(res))
