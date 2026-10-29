import numpy as np, sys, json
from scipy.ndimage import maximum_filter, map_coordinates
from affine import run, domain_field, s3
from hexcore import Vhex, nodetype, mismatch
N=32
def cplx(e): return [[e.real,-e.imag],[e.imag,e.real]]
cases={'twist':cplx(1j*4/64),
       'tw+iso1':cplx(1j*3/64+np.sqrt(3)/2/N*1),
       'tw+iso2':cplx(1j*2/64+np.sqrt(3)/2/N*2),
       'tw+uni1':[[1/(N*s3),-4/64],[4/64,0]],
       'tw+uni2':[[2/(N*s3),-4/64],[4/64,0]],
       'tw+uni3':[[3/(N*s3),-4/64],[4/64,0]],
       'iso_only':cplx(np.sqrt(3)/2/N*2),
       'tw+iso3':cplx(1j*1/64+np.sqrt(3)/2/N*3),
       'tw+iso4':cplx(1j*1/64+np.sqrt(3)/2/N*1)}
def measure(R,radii=(1.5,2.5,4.0,6.0)):
    x,y,ux,uy=R['x'],R['y'],R['ux'],R['uy']; dx=x[1]-x[0]; dy=y[1]-y[0]
    V=Vhex(ux,uy,1.)[0]; t=nodetype(ux,uy)
    pk=(V==maximum_filter(V,size=9,mode='wrap'))&(t>1.5)&(V>4)
    f=domain_field(ux,uy)
    I,J=np.nonzero(pk); out=[]
    for i,j in zip(I,J):
        x0,y0=x[i],y[j]; res={}
        for r in radii:
            th=np.linspace(0,2*np.pi,1441)[:-1]
            px=((x0+r*np.cos(th))/dx)%len(x); py=((y0+r*np.sin(th))/dy)%len(y)
            v=map_coordinates(f,[px,py],order=1,mode='grid-wrap')
            s=np.sign(v); idx=np.nonzero(s!=np.roll(s,-1))[0]
            angs=[]
            for k in idx:
                k2=(k+1)%len(th); a0=th[k]; a1=th[k2] if k2>0 else 2*np.pi
                angs.append(a0+(a1-a0)*v[k]/(v[k]-v[k2]))
            res[r]=np.degrees(np.sort(np.mod(angs,2*np.pi)))
        out.append(dict(pos=(x0,y0),ang=res))
    return out
if __name__=="__main__":
    tag=sys.argv[1]; A=np.array(cases[tag],float)
    R=run(A); np.savez(f'aff_{tag}.npz',x=R['x'],y=R['y'],ux=R['ux'],uy=R['uy'])
    M=measure(R); summ={}
    for r in (1.5,2.5,4.0,6.0):
        gaps=[]
        for m in M:
            a=m['ang'][r]
            if len(a)==6: gaps.append(np.diff(np.concatenate([a,[a[0]+360]])))
        summ[r]=(len(gaps),np.round(np.mean(gaps,axis=0),2).tolist() if gaps else None, np.round(np.std(gaps,axis=0).max(),2) if gaps else None)
    # wall directions (absolute) at r=4 for first good node
    first=[m for m in M if len(m['ang'][4.0])==6]
    print(tag,'nodes',len(M),'E',round(R['E'],4),'nit',R['nit'])
    for r,v in summ.items(): print('   r=',r,'n_ok',v[0],'mean gaps',v[1],'max std',v[2])
    if first: print('   wall directions at r=4 (deg):',np.round(first[0]['ang'][4.0],1).tolist())
    json.dump(dict(tag=tag,A=A.tolist(),nodes=[dict(pos=m['pos'],ang={str(k):v.tolist() for k,v in m['ang'].items()}) for m in M]),open(f'ang_{tag}.json','w'))
