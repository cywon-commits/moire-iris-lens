import os,numpy as np,json
from nodedisp import node_positions
s3=np.sqrt(3); n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=((N1+0.5*N2)+1j*(s3/2*N2)).ravel(); zi=L*(-30j); zi=zi[np.abs(zi)<138]; r=np.abs(zi)
def ld(t): return t if os.path.exists(t+'.npz') else t+'_ck'
def ur(t): zz,_=node_positions(ld(t),90,rmax=138); return ((zz-zi)*np.conj(zi)/np.maximum(r,1e-9)).real
U0=ur('inc_a90_e+0.00_V0.005_L150_r60'); Us={e:ur(f'inc_a90_e{e:+.2f}_V0.005_L150_r60') for e in (0.05,-0.05,0.01,-0.01)}
out={'r':[],'U0':[],**{str(e):[] for e in Us}}
print(' r   U0/r   ' + '  '.join(f'dU/eps({e:+.2f})' for e in Us))
for R in sorted(set(np.round(r[r>1]))):
    m=np.abs(r-R)<1; out['r'].append(R); out['U0'].append(U0[m].mean())
    for e in Us: out[str(e)].append((Us[e]-U0)[m].mean()/e)
    print(f'{R:4.0f} {U0[m].mean()/R:+.4f} '+'  '.join(f'{((Us[e]-U0)[m].mean()/e):+9.2f}' for e in Us))
json.dump(out,open('incl_resp_tw.json','w'))
