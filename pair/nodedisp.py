import numpy as np
from scipy.ndimage import map_coordinates
s3=np.sqrt(3.); c1=np.array([0.5,1/(2*s3)]); A=np.array([[1.,0.5],[0.,s3/2]])
def node_positions(f,alpha,th=1/30.,rmax=95.,iters=30):
    D=np.load(f+'.npz') if isinstance(f,str) else f; x=D['x']; h=x[1]-x[0]; kap=th*np.exp(1j*np.radians(alpha))
    def U(z):
        I=(z.real-x[0])/h; J=(z.imag-x[0])/h
        return map_coordinates(D['ux'],[I,J],order=3)+1j*map_coordinates(D['uy'],[I,J],order=3)
    # candidate labels
    n=np.arange(-12,13); N1,N2=np.meshgrid(n,n,indexing='ij'); L=(N1*A[0,0]+N2*A[0,1])+1j*(N1*A[1,0]+N2*A[1,1])
    z=(L/kap).ravel(); keep=np.abs(z)<rmax; z=z[keep]; tgt=(c1[0]+1j*c1[1])+L.ravel()[keep]
    # coarse grid search for a good start
    ux,uy=D['ux'],D['uy']; zs=[]
    for zi,ti in zip(z,tgt):
        i0=int(round((zi.real-x[0])/h)); j0=int(round((zi.imag-x[0])/h)); w=int(10/h)
        a0,a1=max(i0-w,0),min(i0+w+1,len(x)); b0,b1=max(j0-w,0),min(j0+w+1,len(x))
        R=np.abs(ux[a0:a1,b0:b1]+1j*uy[a0:a1,b0:b1]-ti); k=np.unravel_index(np.argmin(R),R.shape)
        zs.append(x[a0+k[0]]+1j*x[b0+k[1]])
    z=np.array(zs)
    for _ in range(iters):
        r=U(z)-tgt; e=0.05
        gx=(U(z+e)-U(z-e))/(2*e); gy=(U(z+1j*e)-U(z-1j*e))/(2*e)
        # solve [Re gx Re gy; Im gx Im gy] dz = r
        a,b,c,d=gx.real,gy.real,gx.imag,gy.imag; det=a*d-b*c
        dxx=(d*r.real-b*r.imag)/det; dyy=(-c*r.real+a*r.imag)/det
        st=dxx+1j*dyy; st=np.where(np.abs(st)>1.,st/np.abs(st),st); z=z-st
    return z, np.abs(U(z)-tgt)
