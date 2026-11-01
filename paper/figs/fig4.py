from style import *
import json,glob
CR=json.load(open(P+'conserv_res.json'))
fig=plt.figure(figsize=(W2,4.3))
g1=fig.add_gridspec(1,3,left=0.07,right=0.985,top=0.94,bottom=0.62,wspace=0.36)
g2=fig.add_gridspec(1,4,left=0.07,right=0.985,top=0.45,bottom=0.075,wspace=0.12,width_ratios=[1.25,1.25,1,1])
# (a)
ax=fig.add_subplot(g1[0])
for k,c,lab in [('mm def (C=4)',CM,'mismatch, area removed'),('mm add (C=-4)','#5dade2','mismatch, area added'),('tw torque -',CT,'twist, torque $-$')]:
    v=CR[k]; r=np.array(v['r']); ax.step(r,v['dQ'],where='mid',c=c,lw=0.5,alpha=0.55); ax.plot(r,v['dQlin'],'-',c=c,lw=1.6,label=lab)
ax.axvspan(36,58,color='k',alpha=0.06,lw=0); ax.axhline(0,c='k',lw=0.4)
ax.set_xlabel(r'loop radius $\rho$'); ax.set_ylabel(r'$\Delta Q(\rho)=Q(\rho)-Q_{\rm bg}(\rho)$')
ax.set_xlim(r[0],r[-1]); ax.set_ylim(-4.2,7.2); ax.legend(loc='upper left',fontsize=6.2)
ax.text(47,-3.9,'window',ha='center',va='bottom',fontsize=6.2,color='#555')
plabel(ax,'(a)','far charge around one defect')
# (b)
ax=fig.add_subplot(g1[1]); Pp=[];Mm=[]
for k,v in CR.items():
    r=np.array(v['r']); m=(r>=36)&(r<58); p=np.mean(np.array(v['dQlin'])[m]); q=np.mean(np.array(v['dQ'])[m]); Pp.append(p); Mm.append(q)
    mk='o' if v['fam']=='single' else 's'; c=CM if k.startswith('mm') else CT
    ax.scatter(p,q,marker=mk,s=18,facecolor=c,edgecolor='k',lw=0.35,zorder=3)
Pp=np.array(Pp);Mm=np.array(Mm); sl=np.sum(Pp*Mm)/np.sum(Pp*Pp); cc=np.corrcoef(Pp,Mm)[0,1]
xx=np.array([-2.4,1.6]); ax.plot(xx,xx,'k-',lw=0.5)
ax.scatter([],[],marker='o',s=18,facecolor='w',edgecolor='k',label='single defects'); ax.scatter([],[],marker='s',s=18,facecolor='w',edgecolor='k',label='pairs')
ax.set_xlim(-2.4,1.6); ax.set_ylim(-2.4,1.6)
ax.set_xlabel(r'prediction $-\frac{2\pi}{A_c}{\rm Re}(\bar\kappa\sum C)$'); ax.set_ylabel(r'measured $\langle\Delta Q\rangle_{36<\rho<58}$')
ax.legend(loc='upper left'); ax.text(0.97,0.04,f'slope {sl:.2f}, corr. {cc:.2f}',transform=ax.transAxes,ha='right',fontsize=6.8)
plabel(ax,'(b)','conservation of far charge')
# (c)
O=json.load(open(P+'opp_results.json')); d=np.array([o['d'] for o in O])/RL
ax=fig.add_subplot(g1[2])
ax.plot(d,[o['U0pm'] for o in O],'o-',c='#8e44ad',ms=3.5,label=r'elastic only, $(+,-)$')
ax.plot(d,[o['U0pp'] for o in O],'s--',c='#555555',ms=3.5,label=r'elastic only, $(+,+)$')
ax.plot(d,[o['Upm0'] for o in O],'o-',c='#8e44ad',mfc='w',ms=3.5,label=r'moiré-mediated, $(+,-)$')
ax.plot(d,[o['Upp0'] for o in O],'s--',c='#555555',mfc='w',ms=3.5,label=r'moiré-mediated, $(+,+)$')
ax.axhline(0,c='k',lw=0.4); ax.set_xlabel(r'separation $d/R$'); ax.set_ylabel('pair energy $U(d)$')
ax.legend(loc='lower right',fontsize=6.0); ax.set_ylim(-5.6,2.3)
plabel(ax,'(c)','mismatch pair energies')
# (d) twist branch landscape
J=lambda t: json.load(open(P+t+'.json'))['E']; L='_lam-0.5'
ds=[8.,16.,21.,26.,30.,34.,40.,48.,56.]; bg=J('none_d0.0_V0.005'+L)
U0={dd:J(f'pair_d{dd}_V0.0_a0_x'+L)-2*J(f'R_d{dd}_V0.0_a0_x'+L)+J('none_d0.0_V0.0_a0_x'+L) for dd in ds}
Uof=lambda dd,E: E-2*J(f'R_d{dd}_V0.005'+L)+bg-U0[dd]
br={dd:[J(f'pair_d{dd}_V0.005'+L)] for dd in ds}
for dd in ds:
    for f in glob.glob(P+f'pair_d{dd}_V0.005{L}_from_*.json'): br[dd].append(json.load(open(f))['E'])
    for s_ in ['out','in']:
        if os.path.exists(P+f'sw{s_}_d{dd}.json'): br[dd].append(J(f'sw{s_}_d{dd}'))
TB=json.load(open(P+'twist_branches.json'))
ax=fig.add_subplot(g2[0:2])
for dd in ds:
    E=[Uof(dd,e) for e in br[dd]]; ax.plot([dd/RL]*len(E),E,'o',c='#c8c8c8',ms=3.2,zorder=1)
o=np.array(TB['out']); o=o[np.argsort(o[:,0])]; i_=np.array(TB['inn']); i_=i_[np.argsort(i_[:,0])]; env=np.array(TB['env'])
ax.plot(o[:,0]/RL,o[:,1],'>-',c='#d35400',ms=3.5,label='pulled apart (seeded from smaller $d$)')
ax.plot(i_[:,0]/RL,i_[:,1],'<-',c='#1e8449',ms=3.5,label='pushed together (seeded from larger $d$)')
ax.plot(env[:,0]/RL,env[:,1],'k-',lw=2.2,alpha=0.35,label='lowest branch')
ax.plot([],[],'o',c='#c8c8c8',ms=3.2,label='all converged branches')
ax.axhline(0,c='k',lw=0.4); ax.set_xlabel(r'separation $d/R$'); ax.set_ylabel('pair energy $U(d)$'); ax.set_ylim(-2.1,6.4)
ax.legend(loc='upper left',ncol=2,fontsize=6.0,columnspacing=0.8)
plabel(ax,'(d)','twist $(+,+)$ pair: integer branches and hysteresis')
# (e,f) rotation maps
def wmap(ax,f,t):
    D=np.load(P+f+'.npz'); x=D['x']; h=x[1]-x[0]; w=0.5*(np.gradient(D['uy'],h,axis=0)-np.gradient(D['ux'],h,axis=1)); w[D['core']]=np.nan
    m=np.abs(x)<54; ax.imshow(w[np.ix_(m,m)].T,origin='lower',extent=[x[m][0],x[m][-1]]*2,cmap='RdBu_r',vmin=-.3,vmax=.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,loc='left',pad=3,fontsize=8)
wmap(fig.add_subplot(g2[2]),'swout_d40.0','(e) pulled, $d=40$')
wmap(fig.add_subplot(g2[3]),'swin_d40.0','(f) pushed, $d=40$')
save(fig,'fig4')
print('slope',sl,'corr',cc,len(Pp))
