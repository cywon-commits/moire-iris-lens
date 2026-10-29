import numpy as np
def images(beta,a,m,tol=1e-7):
    c=[a*(abs(a)**2-1), beta*(1-2*abs(a)**2)+a*np.conj(beta), m+np.conj(a)*(beta**2-2*a*m)-np.conj(m)-abs(beta)**2,
       2*np.conj(a)*beta*m-m*np.conj(beta), np.conj(a)*m**2]
    while len(c) and abs(c[0])<1e-14: c=c[1:]
    if len(c)<2: return []
    out=[]
    for w in np.roots(c):
        if abs(w)<1e-9: continue
        z=(-a*w**2+beta*w+m)/w
        if abs(w-np.conj(z))<tol*max(1,abs(z)) and abs(z+a*np.conj(z)-m/np.conj(z)-beta)<1e-6*max(1,abs(beta)):
            par=1 if 1-abs(a+m/w**2)**2>0 else -1
            if all(abs(z-q[0])>1e-6 for q in out): out.append((z,par))
    return out
def count_grid(a,m,L=3.,n=301):
    b=np.linspace(-L,L,n); N=np.zeros((n,n),int); D=np.zeros((n,n),int)
    for i,x in enumerate(b):
        for j,y in enumerate(b):
            im=images(x+1j*y,a,m); N[i,j]=len(im); D[i,j]=sum(p for _,p in im)
    return b,N,D
if __name__=="__main__":
    from caustic import critical
    for name,phim in [('converging',0.),('twist (vacancy)',-90.),('diverging',180.)]:
        m=np.exp(1j*np.radians(phim))
        for a in [0.,0.2,0.5,0.5j,0.5*np.exp(0.25j*np.pi)]:
            b,N,D=count_grid(a,m,n=121)
            u,cn=np.unique(N,return_counts=True)
            print(f'{name:16s} a={a:.2f}: image-count areas '+', '.join(f'{v}:{c}' for v,c in zip(u,cn))+f'   n+ - n- values {sorted(set(D[N>0].tolist()))}')
