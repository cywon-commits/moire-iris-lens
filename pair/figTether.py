import os
import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','hex')); from hexcore import Vhex
from wall1d import gamma
J=lambda t: json.load(open(t+'.json'))['E']; s3=np.sqrt(3)
fig=plt.figure(figsize=(16,9.5))
# (a) 1D wall tension
ax=fig.add_subplot(2,3,1); chis=np.linspace(0,90,7)
for lam,c in [(1.,'#c0392b'),(-0.5,'#2471a3')]:
    g=[gamma(np.radians(x),lam) for x in chis]; ax.plot(chis,g,'o',c=c,label=f'1D relaxed wall, $K/\\mu={1+lam:g}$')
    cc=np.linspace(0,90,100); ax.plot(cc,g[-1]*np.sqrt(1+(1+lam)*np.cos(np.radians(cc))**2),'-',c=c,lw=1,alpha=.7)
ax.set_xlabel(r'angle $\chi$ between Burgers vector $\mathbf{b}$ and wall normal'); ax.set_ylabel(r'line tension $\gamma$')
ax.text(3,0.072,'tensile',fontsize=9); ax.text(72,0.04,'shear',fontsize=9)
ax.set_title(r'(a) $\gamma(\chi)=\gamma_s\sqrt{1+(1+\lambda/\mu)\cos^2\chi}$  (lines)'+'\nshear wall K-blind; tensile wall ∝ K',fontsize=10); ax.legend(fontsize=8)
# (b) pull-push gap
ax=fig.add_subplot(2,3,2); ds=[16.,21.,30.,34.,40.,48.]
g5=[J(f'swout_d{d}')-J(f'swin_d{d}') for d in ds]; g2=[J(f'twL1out_d{d}')-J(f'twL1in_d{d}') for d in ds]
ax.plot(ds,g5,'o-',c='#2471a3',label=r'$K/\mu=0.5$'); ax.plot(ds,g2,'s-',c='#c0392b',label=r'$K/\mu=2$')
ax.plot(ds,np.array(g5)*0.0706/0.0469,'--',c='gray',label='tensile-tether prediction (×1.51)')
ax.annotate('tether snapped\nat K/μ=0.5 only',(48,g5[-1]),(36,6.2),arrowprops=dict(arrowstyle='->'),fontsize=8.5)
ax.set_xlabel('d'); ax.set_ylabel(r'$E_{\rm pull}-E_{\rm push}$'); ax.set_title('(b) twist: pulled-branch excess energy\n(same core positions → single-defect parts cancel)',fontsize=10); ax.legend(fontsize=8)
# (c) decomposition
ax=fig.add_subplot(2,3,3)
parts={'dilation\n(μ+λ)tr²/2':0.265,'deviatoric\n(shear)':2.011,'registry V':1.289,'core rim\n(r<5.4)':4.253-3.565}
ax.bar(range(4),list(parts.values()),color=['#e67e22','#7d3c98','#555','#bbb']); ax.set_xticks(range(4)); ax.set_xticklabels(parts.keys(),fontsize=8.5)
ax.set_ylabel('excess energy (pull − push), d=40, K/μ=0.5'); ax.set_title('(c) where the excess lives: almost no dilation\n→ not a tensile tether; distorted shear network',fontsize=10)
# (d) mismatch branches
ax=fig.add_subplot(2,3,4)
lab=['d=21','d=26']; free=[J('mmout_d21.0')-J('mmin_d21.0'),J('mmout_d26.0')-J('mmin_d26.0')]
sym=[J('mmout_d21.0_symz')-J('mmin_d21.0_symz'),J('mmout_d26.0_symz')-J('mmin_d26.0_symz')]
x=np.arange(2); ax.bar(x,free,.35,color='#2471a3',label='free'); ax.bar(x+.35,sym,.35,color='#85c1e9',label='mirror-constrained (re-relaxed)')
ax.set_xticks(x+.17); ax.set_xticklabels(lab); ax.axhline(0,c='k',lw=.6); ax.set_ylabel(r'$E_{\rm pull}-E_{\rm push}$'); ax.legend(fontsize=8)
ax.set_title('(d) mismatch (K/μ=0.5): branch gap\nd=21 removed by mirror (chirality); d=26 survives (structural)',fontsize=10)
def vmap(ax,f,t):
    D=np.load(f+'.npz'); x=D['x']; m=np.abs(x)<34; V,_,_=Vhex(D['ux'],D['uy'],0.005,s3*0.005); V[D['core']]=np.nan
    ax.imshow(V[np.ix_(m,m)].T,origin='lower',extent=[x[m][0],x[m][-1]]*2,cmap='magma'); ax.set_title(t,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
vmap(fig.add_subplot(2,3,5),'mmout_d26.0',f'(e) mismatch d=26, pulled branch  E={J("mmout_d26.0"):.2f}\nwalls wired core–node–core along the axis')
vmap(fig.add_subplot(2,3,6),'mmin_d26.0',f'(f) mismatch d=26, pushed branch  E={J("mmin_d26.0"):.2f}\ncentral node wired vertically instead')
plt.tight_layout(); plt.savefig('tether_test.png',dpi=120)
