import numpy as np, json
from hexcore import s3
def delaunay_dirs(A):
    Ai=np.linalg.inv(np.array(A,float)); vs=[]
    for n1 in range(-4,5):
        for n2 in range(-4,5):
            if n1==0 and n2==0: continue
            vs.append(-Ai@np.array([n1+0.5*n2,n2*s3/2]))
    vs=sorted(vs,key=lambda v: v@v); v1=vs[0]
    v2=next(v for v in vs if abs(v1[0]*v[1]-v1[1]*v[0])>1e-9)
    v3=min([v1+v2,v1-v2],key=lambda v: v@v)
    ds=[v1,v2,v3,-v1,-v2,-v3]
    return np.sort([np.degrees(np.arctan2(v[1],v[0]))%360 for v in ds]), [np.hypot(*v) for v in (v1,v2,v3)]
def wrap(a): return (a+180)%360-180
if __name__=="__main__":
    from angles import cases
    rows=[]
    for tag in ['twist','tw+iso1','tw+iso2','tw+iso4','tw+iso3','iso_only','tw+uni1','tw+uni2','tw+uni3']:
        A=np.array(cases[tag]); s=A[0,0]; th=A[1,0]; chi=np.degrees(np.arctan2(s,th)) if tag.find('uni')<0 else None
        ch,L=delaunay_dirs(A); d=json.load(open(f'ang_{tag}.json'))
        out={}
        for r in ['1.5','2.5','4.0','6.0']:
            dev=[];gaps=[]
            for nd in d['nodes']:
                a=np.array(nd['ang'][r])
                if len(a)!=6: continue
                dev.append([wrap(ai-ch[np.argmin(abs(wrap(ai-ch)))]) for ai in a])
                gaps.append(np.diff(np.concatenate([np.sort(a),[np.sort(a)[0]+360]])))
            dev=np.array(dev); out[r]=(np.mean(dev),np.mean(dev,axis=0))
        print(f"{tag:9s} chi={'%5.1f'%chi if chi is not None else '  -  '}  moire L={np.round(L,1)}  chords {np.round(ch,1)}")
        print('          mean rotation vs chord (deg): '+'  '.join(f"r={r}: {out[r][0]:+5.1f}" for r in out))
        if 'uni' in tag: print('          per-wall deviation r=2.5:',np.round(out['2.5'][1],1),' r=6:',np.round(out['6.0'][1],1))
        rows.append(dict(tag=tag,chi=chi,dev={r:out[r][0] for r in out},perwall={r:out[r][1].tolist() for r in out},chords=ch.tolist()))
    json.dump(rows,open('spiral.json','w'))
