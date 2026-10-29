import os
import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
s3=np.sqrt(3); R=json.load(open('cdef_topo.json')); Rl=np.sqrt(120.)
by={r['f']:r for r in R}
rows=[('mismatch α=0',['c_mm_def','c_mm_tqp','c_mm_tqm','c_mm_add'],['area removed','torque +','torque −','area added']),
      ('twist α=90°',['c_tw_tqp','c_tw_def','c_tw_add','c_tw_tqm'],['torque + (with twist)','area removed','area added','torque − (against twist)'])]
fig=plt.figure(figsize=(19,8.6))
for i,(rt,fs,labs) in enumerate(rows):
    for j,(f,l) in enumerate(zip(fs,labs)):
        ax=fig.add_subplot(2,5,5*i+j+1); D=np.load(f+'.npz'); x=D['x']; m=(np.abs(x-0)<44)
        V,_,_=Vhex(D['ux'],D['uy'],0.005,s3*0.005); V[D['core']]=np.nan
        ax.imshow(V[np.ix_(m,m)].T,origin='lower',cmap='magma',extent=[x[m][0],x[m][-1]]*2)
        r=by[f]; rs=2*Rl*abs(np.sin(np.radians(r['phim'])/2))
        if rs>0: ax.add_patch(plt.Circle((8,0),rs,fill=False,ec='c',ls='--',lw=1))
        ax.set_title(f"{rt}: {l}\nφm={r['phim']:.0f}°  q={r['absorbed']:+d} (pred {r['pred']:.2f})  z={r['z7']}",fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(2,5,5); ph=np.linspace(-180,180,200)
ax.plot(ph,2*np.pi*4/30/(s3/2)*(1-np.cos(np.radians(ph))),'k-',lw=1,label=r'lens: $2\pi\theta|C|(1-\cos\phi_m)/A_c$')
for r in R:
    c='#2471a3' if r['alpha']==0 else '#c0392b'; ax.scatter(r['phim'],r['absorbed'],s=60,color=c,edgecolor='k',zorder=3)
ax.scatter([],[],color='#2471a3',label='mismatch'); ax.scatter([],[],color='#c0392b',label='twist')
ax.set_xlabel(r'$\phi_m=\arg C-\alpha$ (deg)'); ax.set_ylabel('absorbed node charge q'); ax.set_xticks([-180,-90,0,90,180]); ax.legend(fontsize=8)
ax.set_title('(i) locally absorbed charge vs mass phase',fontsize=10)
ax=fig.add_subplot(2,5,10)
for r in R:
    c='#2471a3' if r['alpha']==0 else '#c0392b'; ax.scatter(abs(r['phim']),r['dE'],s=60,color=c,edgecolor='k')
    ax.annotate(r['f'][2:],(abs(r['phim']),r['dE']),fontsize=7,xytext=(3,3),textcoords='offset points')
ax.set_xlabel(r'$|\phi_m|$ (deg)'); ax.set_ylabel('defect energy E − E_bg'); ax.set_xticks([0,90,180])
ax.set_title('(j) defect energy rises with |φm|',fontsize=10)
plt.tight_layout(); plt.savefig('area_add_torque.png',dpi=110)
