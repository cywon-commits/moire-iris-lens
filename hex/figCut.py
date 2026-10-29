import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hexcore import Vhex
plt.rcParams.update({'font.size':10})
fig,ax=plt.subplots(1,4,figsize=(18,5))
pan=[('ten_lam_1.0.npz',0,'(a) no defect: two 6-wall nodes,\none connecting wall'),
     ('ten_cutp_2.0.npz',2,'(b) extra row ℓ=2a, b=+a₁:\nnearer node is pulled in'),
     ('ten_cutp_8.0.npz',8,'(c) ℓ=8a, b=+a₁: that node is consumed;\nits walls end on the 2 dislocations'),
     ('ten_cutm_8.0.npz',8,'(d) ℓ=8a, b=−a₁: nodes repelled,\ncut wrapped by a closed wall')]
for a,(f,l,t) in zip(ax,pan):
    D=np.load(f); x=D['x']; V=Vhex(D['ux'],D['uy'],1.,np.sqrt(3))[0]
    a.imshow(V.T,origin='lower',extent=[x[0],x[-1],x[0],x[-1]],cmap='magma',vmin=0,vmax=9)
    if l: a.plot([0,0],[-l/2,l/2],color='#5ad',lw=2.2); a.plot([0,0],[-l/2,l/2],'o',color='#5ad',ms=5)
    a.set_title(t,fontsize=10); a.set_xticks([-16,0,16]); a.set_yticks([-16,0,16])
fig.text(0.5,0.01,'blue line: eigen-slip (extra half-row segment) with Burgers ±a₁ across it; blue dots: its two edge dislocations',ha='center',fontsize=9,color='#444')
fig.savefig('cut_results.png',dpi=110,bbox_inches='tight')
