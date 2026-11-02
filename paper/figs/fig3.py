from style import *
import json
from matplotlib.patches import Circle
R=json.load(open(P+'cdef_topo.json')); by={r['f']:r for r in R}
CR=json.load(open(P+'conserv_res.json')); far={v['file']:v for v in CR.values() if v['fam']=='single'}
rows=[('mismatch',['c_mm_tqm','c_mm_def','c_mm_tqp','c_mm_add']),('twist',['c_tw_def','c_tw_tqp','c_tw_add','c_tw_tqm'])]
lab={'def':'area removed','add':'area added','tqp':'torque +','tqm':'torque −'}
fig=plt.figure(figsize=(W2,3.05))
gs=fig.add_gridspec(2,5,width_ratios=[1,1,1,1,0.12],left=0.035,right=0.655,bottom=0.03,top=0.855,wspace=0.06,hspace=0.32)
lim=34
for i,(fam,fs) in enumerate(rows):
    for j,f in enumerate(fs):
        ax=fig.add_subplot(gs[i,j]); D=np.load(P+f+'.npz'); x=D['x']; V=Vmap(D)
        mx=np.abs(x-8)<lim+1; my=np.abs(x)<lim+1
        ax.imshow(logV(V[np.ix_(mx,my)]).T,origin='lower',extent=[x[mx][0],x[mx][-1],x[my][0],x[my][-1]],interpolation='bilinear',**LOGV)
        ax.add_patch(Circle((8,0),RL,fill=False,ec='w',ls=':',lw=0.7))
        ax.set_xlim(8-lim,8+lim); ax.set_ylim(-lim,lim); ax.set_xticks([]); ax.set_yticks([])
        r=by[f]; t=f.split('_')[2]
        ax.set_title(f"{lab[t]}\n$\\varphi_m$={str(int(r['phim'])).replace('-','−')}°, $q$={r['absorbed']:d}",fontsize=6.8,pad=2,linespacing=1.15)
        if j==0: ax.text(-0.1,0.5,fam,transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=8)
fig.text(0.035,0.955,'(a) eight complex defects, $|C|=4$ (dotted: Einstein radius $R$)',fontsize=8,ha='left',va='bottom')
ax=fig.add_axes([0.735,0.37,0.255,0.485])
ph=np.linspace(-180,180,361); T=2*np.pi*TH*4/AC
ax.plot(ph,T*(1-np.cos(np.radians(ph))),'-',c='k',lw=1.0,label=r'$T(1-\cos\varphi_m)$')
ax.plot(ph,T*np.cos(np.radians(ph)),'--',c='k',lw=1.0,label=r'$T\cos\varphi_m$')
ax.axhline(T,c=CG,lw=0.6,ls=':'); ax.text(-160,T-0.06,r'$T$',fontsize=7,color=CG,va='top',ha='left')
for r in R:
    c=CM if r['alpha']==0 else CT; mk='o' if r['alpha']==0 else 's'
    ax.scatter(r['phim'],r['absorbed'],s=22,marker=mk,facecolor=c,edgecolor='k',lw=0.4,zorder=3)
    v=far[r['f']]; rr=np.array(v['r']); m=(rr>=36)&(rr<58); dq=np.mean(np.array(v['dQ'])[m])
    ax.scatter(r['phim'],-dq,s=22,marker=mk,facecolor='w',edgecolor=c,lw=0.9,zorder=3)
ax.scatter([],[],s=22,marker='o',facecolor='k',edgecolor='k',label='local $q$')
ax.scatter([],[],s=22,marker='o',facecolor='w',edgecolor='k',label=r'far $-\Delta Q$')
ax.scatter([],[],s=22,marker='o',facecolor=CM,edgecolor='k',lw=0.4,label='mismatch'); ax.scatter([],[],s=22,marker='s',facecolor=CT,edgecolor='k',lw=0.4,label='twist')
ax.set_xticks([-180,-90,0,90,180]); ax.set_xlim(-190,190); ax.set_ylim(-1.45,2.75)
ax.set_xlabel(r'$\varphi_m=\arg C-\alpha$ (deg)'); ax.set_ylabel('node charge')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.24),ncol=3,fontsize=6.0,handletextpad=0.2,columnspacing=0.6)
ax.set_title('(b) local + far $=T$',loc='left',pad=3,fontsize=8)
save(fig,'fig3')
