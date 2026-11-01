from style import *
import json
from collections import Counter
R=json.load(open(P+'topo_all.json')); ZW=json.load(open(P+'zw.json'))
fig,axs=plt.subplots(1,3,figsize=(W2,2.2),gridspec_kw=dict(wspace=0.38,left=0.075,right=0.99,bottom=0.19,top=0.88))
ax=axs[0]
for fam,c,mk,off in (('twist',CT,'o',-0.12),('mismatch',CM,'s',0.12)):
    X=[sum(cc['Qhole'] for cc in r['cores'])-r['Nminus'] for r in R if r['fam']==fam]
    Y=[r['Qbox']-r['Nplus'] for r in R if r['fam']==fam]
    Cn=Counter(zip(X,Y))
    for (x_,y_),n in Cn.items(): ax.scatter(x_+off,y_,s=10+6*n,marker=mk,facecolor='none',edgecolor=c,lw=0.8)
    ax.scatter([],[],s=18,marker=mk,facecolor='none',edgecolor=c,lw=0.8,label=f'{fam} ({len(X)} states)')
ax.plot([-1.5,4.5],[-1.5,4.5],'k-',lw=0.5,zorder=0); ax.set_xlim(-1.5,4.5); ax.set_ylim(-1.5,4.5)
ax.set_xlabel(r'bound charge $\sum_k Q(\partial H_k)-N_-$'); ax.set_ylabel(r'missing ordinary nodes $Q(\partial D)-N_+$')
ax.legend(loc='upper left'); ax.text(0.97,0.04,'all on the line (deviation 0)',transform=ax.transAxes,ha='right',fontsize=6.8)
plabel(ax,'(a)','Gauss law for node charge')
ax=axs[1]
for fam,off,c in (('twist',-0.18,CT),('mismatch',0.18,CM)):
    q=[p['Qhole']-p['Nminus_local'] for r in R if r['fam']==fam and r['mode'] in ('R','pair') for p in r['cores']]
    C=Counter(q); xs=sorted(C); ax.bar(np.array(xs)+off,[C[x] for x in xs],0.36,color=c,label=f'{fam} ({len(q)} cores)')
ax.set_xlabel(r'local charge per core $q=Q_{\rm hole}-N_-$'); ax.set_ylabel('count'); ax.legend(loc='upper right'); ax.set_ylim(0,95)
ax.set_xticks(range(-2,4)); ax.set_xlim(-2.6,3.6)
plabel(ax,'(b)','integer charge bound to a core')
ax=axs[2]
Lm={0:0,1:3,2:5,3:6,4:7,5:8,6:9,7:9}
C=Counter((abs(o['w']),o['z']) for o in ZW)
for (w,z),n in C.items(): ax.scatter(w,z,s=4+1.2*n,facecolor=CG,edgecolor='k',lw=0.3,alpha=0.85,zorder=3)
ww=np.arange(0,8); ax.step(ww,[2*Lm[w] for w in ww],where='mid',c=CT,lw=1.1,label=r'bound $z=2L_{\min}(|w|)$')
ax.scatter([],[],s=20,facecolor=CG,edgecolor='k',lw=0.3,label='circles (area $\\propto$ count)')
ax.set_xlabel(r'AA winding $|w|$ inside the circle'); ax.set_ylabel(r'walls crossed $z$')
ax.set_yticks(range(0,29,4)); ax.set_ylim(-1,29); ax.set_xlim(-0.5,10.3); ax.set_xticks([0,2,4,6,8,10]); ax.legend(loc='lower right',title=f'{len(ZW)} circles, 0 violations',title_fontsize=6.8)
plabel(ax,'(c)','wall valence vs winding')
save(fig,'fig2')
print(max(abs(r['Qbox']-(r['Nplus']-r['Nminus']+sum(cc['Qhole'] for cc in r['cores']))) for r in R), max(o['z'] for o in ZW), max(abs(o['w']) for o in ZW))
