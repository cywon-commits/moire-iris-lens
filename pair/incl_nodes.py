import json,numpy as np
from nodedisp import node_positions
import sys
r0=float(sys.argv[1]); eps=float(sys.argv[2]); Lp=int(sys.argv[3]); fams=[(int(a),n) for a,n in [q.split(':') for q in sys.argv[4:]]]; out={}
T=lambda a,e,V: f'inc_a{a}_e{e:+.2f}_V{V}_L{Lp}_r{r0:.0f}'
zb,eb=node_positions(T(0,eps,0.0),0,rmax=Lp-12); zb0=np.array(zb)
# bare reference: background nodes for V0=0 are exactly L/kap
th=1/30.; 
def ref_nodes(alpha):
    s3=np.sqrt(3); n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=(N1+0.5*N2)+1j*(s3/2*N2); kap=th*np.exp(1j*np.radians(alpha)); z=(L/kap).ravel(); return z[np.abs(z)<Lp-12]
Ub=zb-ref_nodes(0); rb=np.abs(ref_nodes(0)); Urb=(Ub*np.conj(ref_nodes(0))/rb).real
bins=np.arange(r0+6,Lp-12,8.)
def binned(r,v):
    return np.array([np.mean(v[(r>=b)&(r<b+8)]) if np.any((r>=b)&(r<b+8)) else np.nan for b in bins])
gb=binned(rb[rb>r0+6],(Urb*rb)[rb>r0+6])   # r*U_r for bare (= a_bare * shape)
out['bins']=(bins+4).tolist(); out['bare_rUr']=gb.tolist()
for a,n in fams:
    zp,ep=node_positions(T(a,eps,0.005),a,rmax=Lp-12); zm,em=node_positions(T(a,-eps,0.005),a,rmax=Lp-12)
    ok=(ep<1e-3)&(em<1e-3); Ua=(zp-zm)/2; z0=(zp+zm)/2; r=np.abs(z0); Ur=(Ua*np.conj(z0)/r).real; Ut=(Ua*np.conj(z0)/r).imag
    m=ok&(r>r0+6)
    S=binned(r[m],(Ur*r)[m])/gb*(1/3)
    out[n]=dict(S=S.tolist(),r=r[m].tolist(),rUr=(Ur*r)[m].tolist(),rUt=(Ut*r)[m].tolist(),nok=int(ok.sum()))
    print(n,'nodes ok',ok.sum(),'/',len(ok),' S(rho)=',np.round(S,3),' mean(rU_t)=',round(float(np.mean((Ut*r)[m])),3))
print('bare r*U_r',np.round(gb,2),' (a_bare=S_bare*eps*r0^2=',round(eps*r0**2/3,2),')')
json.dump(out,open(f'incl_nodes_r{r0:.0f}.json','w'))
