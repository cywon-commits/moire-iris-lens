import json,os,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from nodedisp import node_positions
th=1/30.; fit=json.load(open('permod2_fit.json')); s3=np.sqrt(3)
def ld(t): return t if os.path.exists(t+'.npz') else t+'_ck'
n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=((N1+0.5*N2)+1j*(s3/2*N2)).ravel()
def nodes_U(t,a,rm):
    k=np.exp(1j*np.radians(a))/30; zi=(L/k); zi=zi[np.abs(zi)<rm]; zz,_=node_positions(ld(t),a,rmax=rm); r=np.abs(zi)
    ur=((zz-zi)*np.conj(zi)/np.maximum(r,1e-9)).real; return r,ur
fig,axs=plt.subplots(1,4,figsize=(22,5))
cols={'d':'#1f77b4','w':'#d62728','g1':'#2ca02c','g2':'#9467bd'}
lab={'d':'dilation Re U_a','w':'rotation Im U_a','g1':'shear Re U_b','g2':'shear Im U_b'}
for ax,a,t in ((axs[0],90,'twist (α=90°)'),(axs[1],0,'mismatch (α=0°)')):
    D=json.load(open(f'permod2_a{a}.json')); e0=[x['e'] for x in D if x['Ua']==[0,0] and x['Ub']==[0,0]][0]; F=fit[str(a)]
    for m in cols:
        pts=[]
        for x in D:
            v={'d':x['Ua'][0],'w':x['Ua'][1],'g1':x['Ub'][0],'g2':x['Ub'][1]}
            if all(abs(v[k])<1e-12 for k in v if k!=m): pts.append((v[m],x['e']))
        pts=np.array(sorted(pts)); lin=F['sig_d']*th**2*pts[:,0] if m=='d' else 0
        ax.plot(pts[:,0],(pts[:,1]-e0-lin)/th**2,'o-',color=cols[m],label=lab[m]+(' (minus linear term)' if m=='d' else ''),ms=5)
    ax.set_title(f"{t}: periodic cells\nK_M={F['K']:+.3f}θ²  μ_M={F['mu']:.3f}θ²  χ_M={F['chi']:+.2f}θ²  σ_d={F['sig_d']:+.2f}θ²",fontsize=10)
    ax.set_xlabel('moiré displacement-gradient amplitude'); ax.set_ylabel('(e − e₀)/θ²'); ax.axhline(0,color='k',lw=.5); ax.legend(fontsize=8)
def shell(r,u):
    Rs=sorted(set(np.round(r[r>1]))); return np.array(Rs),np.array([u[np.abs(r-R)<1].mean() for R in Rs])
ax=axs[2]
for t,a,Lb,c in (('none_d0.0_V0.005_lam-0.5',90,72,'#d62728'),('/home/claude/ridge/big/B_bg_tw',90,148.5,'#d62728'),('none_d0.0_V0.005_a0_x_lam-0.5',0,72,'#1f77b4'),('/home/claude/ridge/big/B_bg_mm',0,148.5,'#1f77b4'),('inc_a90_e+0.00_V0.005_L150_r60',90,150,'#ff9896')):
    try:
        r,ur=nodes_U(t,a,Lb-10); R,Us=shell(r,ur); ax.plot(R,Us/R,'o-',color=c,ms=5,ls='-' if Lb!=72 else '--',label=f"{'twist' if a==90 else 'mismatch'} box ±{Lb}")
    except Exception as ex: print(ex)
ax.set_xlabel('node radius r'); ax.set_ylabel('U_r / r  (bulk moiré dilation)'); ax.set_title('defect-free clamped boxes: bulk moiré expands (∝1/L)\nexcess twist/strain is pushed into the clamped edge layer',fontsize=10); ax.legend(fontsize=8); ax.axhline(0,color='k',lw=.5)
ax=axs[3]
rb,ub=nodes_U('inc_a0_e+0.05_V0.0_L150_r60',0,138); 
for R in sorted(set(np.round(rb))): pass
def shell(r,u):
    Rs=sorted(set(np.round(r[r>1]))); return np.array(Rs),np.array([u[np.abs(r-R)<1].mean() for R in Rs])
Rb,Ubs=shell(rb,ub); ax.plot(Rb,Ubs/0.05,'k^-',label='bare elastic mismatch (S=1/3 exact)')
try:
    r0_,u0_=nodes_U('inc_a90_e+0.00_V0.005_L150_r60',90,138); R0,U0=shell(r0_,u0_)
    for e,c in ((0.05,'#d62728'),(-0.05,'#ff9896'),(0.01,'#8c564b'),(-0.01,'#c49c94')):
        r,u=nodes_U(f'inc_a90_e{e:+.2f}_V0.005_L150_r60',90,138); R,U=shell(r,u); ax.plot(R,(U-U0)/e,'o-',color=c,label=f'twist ε*={e:+.2f}')
except Exception as ex: print(ex)
ax.plot(Rb,Ubs/0.05*3*fit['90']['S_cauchy'],'r--',label=f"twist linear bulk prediction (S={fit['90']['S_cauchy']:+.2f})")
ax.axvline(60,color='gray',ls=':'); ax.axhline(0,color='k',lw=.5)
ax.set_xlabel('node radius r'); ax.set_ylabel('ΔU_r / ε*'); ax.set_title('moiré eigen-dilation disk r₀=60 in clamped box ±150\n(response relative to the ε*=0 state)',fontsize=10); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig('moire_moduli.png',dpi=105)
