import json,glob,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J=lambda t: json.load(open(t+'.json'))['E']; R=np.sqrt(120.); L='_lam-0.5'
ds=[8.,16.,21.,26.,30.,34.,40.,48.,56.]
bg=J('none_d0.0_V0.005'+L)
U0={d:J(f'pair_d{d}_V0.0_a0_x'+L)-2*J(f'R_d{d}_V0.0_a0_x'+L)+J('none_d0.0_V0.0_a0_x'+L) for d in ds}
Uof=lambda d,E: E-2*J(f'R_d{d}_V0.005'+L)+bg-U0[d]
br={d:[] for d in ds}
for d in ds:
    br[d].append(('default',J(f'pair_d{d}_V0.005'+L)))
    for f in glob.glob(f'pair_d{d}_V0.005{L}_from_*.json'): br[d].append(('seed',J(f[:-5])))
    for s in ['out','in']:
        if glob.glob(f'sw{s}_d{d}.json'): br[d].append((s,J(f'sw{s}_d{d}')))
br[8.].append(('out',J('pair_d8.0_V0.005'+L))); br[56.].append(('in',J('pair_d56.0_V0.005'+L)))
env=np.array([(d,min(Uof(d,e) for _,e in br[d])) for d in ds])
cur={s:np.array([(d,Uof(d,e)) for d in ds for k,e in br[d] if k==s]) for s in ['out','in']}
print('envelope',env.round(2).tolist()); print('out',cur['out'].round(2).tolist()); print('in',cur['in'].round(2).tolist())
json.dump(dict(env=env.tolist(),out=cur['out'].tolist(),inn=cur['in'].tolist()),open('twist_branches.json','w'))
mm5=np.array([(d,J(f'pair_d{d}_V0.005_a0_x_sym'+L)-2*J(f'R_d{d}_V0.005_a0_x_sym'+L)+J('none_d0.0_V0.005_a0_x_sym'+L)-U0[d]) for d in [8.,16.,26.,40.,56.]])
fig=plt.figure(figsize=(15,9.4))
ax=fig.add_subplot(2,3,(1,2))
for d in ds:
    for k,e in br[d]: ax.plot(d/R,Uof(d,e),'o',c='#bbbbbb',ms=5,zorder=1)
o=cur['out'][np.argsort(cur['out'][:,0])]; i=cur['in'][np.argsort(cur['in'][:,0])]
ax.plot(o[:,0]/R,o[:,1],'>-',c='#d35400',label='pull apart (seeded from smaller d)')
ax.plot(i[:,0]/R,i[:,1],'<-',c='#1e8449',label='push together (seeded from larger d)')
ax.plot(env[:,0]/R,env[:,1],'k-',lw=2.5,alpha=.6,label='lower envelope of all branches (twist, $K/\\mu=0.5$)')
ax.plot(mm5[:,0]/R,mm5[:,1],'s--',c='#2471a3',label='mismatch lens-only, $K/\\mu=0.5$ (no sweep done)')
ax.text(2.2,3.9,'pull branch slope ≈ 0.11 per length (16→40)',color='#d35400',fontsize=9)
for xv,t in [(np.sqrt(2),r'$\sqrt{2}R$'),(2*np.sqrt(2),r'$2\sqrt{2}R$')]: ax.axvline(xv,c='k',ls=':',lw=.8); ax.text(xv+.03,-1.6,t,fontsize=9)
ax.axhline(0,c='k',lw=.6); ax.set_xlabel('$d/R$'); ax.set_ylabel('$U(d)$'); ax.set_ylim(-1.9,5.2)
ax.set_title('(a) twist pair: multi-branch landscape (grey = every converged branch) and pull/push hysteresis',fontsize=10.5); ax.legend(fontsize=8.5,loc='upper right')
ax=fig.add_subplot(2,3,3)
for d in ds:
    E=[Uof(d,e) for _,e in br[d]]; ax.plot([d/R]*len(E),np.array(E)-min(E),'o',c='#7d3c98',alpha=.6)
ax.set_xlabel('$d/R$'); ax.set_ylabel('branch energy above envelope'); ax.set_title('(b) spread of branches at fixed $d$\n(discrete wall-network topologies)',fontsize=10)
def wmap(ax,f,title):
    D=np.load(f+'.npz'); x=D['x']; h=x[1]-x[0]; w=0.5*(np.gradient(D['uy'],h,axis=0)-np.gradient(D['ux'],h,axis=1)); w[D['core']]=np.nan
    m=np.abs(x)<60; ax.imshow(w[np.ix_(m,m)].T,origin='lower',extent=[x[m][0],x[m][-1]]*2,cmap='RdBu_r',vmin=-.3,vmax=.3)
    ax.set_title(title,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
wmap(fig.add_subplot(2,3,4),'swout_d40.0',f'(c) pulled apart to d=40: U={Uof(40.,J("swout_d40.0")):+.2f}\nwall tether joins the two cores')
wmap(fig.add_subplot(2,3,5),'swin_d40.0',f'(d) pushed in to d=40: U={Uof(40.,J("swin_d40.0")):+.2f}\neach core wired separately')
wmap(fig.add_subplot(2,3,6),'swin_d21.0',f'(e) pushed in to d=21: U={Uof(21.,J("swin_d21.0")):+.2f}\n(envelope branch)')
plt.tight_layout(); plt.savefig('twist_branches.png',dpi=120)
