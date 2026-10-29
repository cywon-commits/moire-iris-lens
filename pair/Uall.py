import json,glob,numpy as np
E={}
for f in glob.glob('*.json'):
    try: d=json.load(open(f))
    except Exception: continue
    if 'mode' not in d: continue
    E[(d['mode'],d['d'],d['V0'],d.get('alpha',90.),d.get('orient','x'))]=d['E']
Ue={d:E[('pair',d,0.0,90.,'x')]-2*E[('R',d,0.0,90.,'x')] for d in [8.,12.,16.,20.,26.,32.,40.,50.] if ('pair',d,0.0,90.,'x') in E}
out={}
for al,ori,ds in [(90.,'x',[8.,12.,16.,20.,26.,32.,40.,50.]),(0.,'x',[8.,16.,26.,40.]),(90.,'y',[12.,26.,40.])]:
    bg=E[('none',0.0,0.005,al,'x')] if ('none',0.0,0.005,al,'x') in E else E[('none',0.0,0.005,90.,'x')]
    rows=[]
    for d in ds:
        k=('pair',d,0.005,al,ori)
        if k not in E: continue
        Rr=E[('R',d,0.005,al,ori)]; L=E.get(('L',d,0.005,al,ori),Rr)
        Uv=E[k]-L-Rr+bg; rows.append((d,Uv-Ue[d],Uv,Ue[d]))
    out[f'{al:.0f}_{ori}']=rows
    print(f'alpha={al:.0f} orient={ori}: '+'  '.join(f'd={r[0]:.0f}: U_moire={r[1]:+.2f}' for r in rows))
print('symmetry check (alpha=0, d=26): L =',E.get(('L',26.,0.005,0.,'x')),' R =',E.get(('R',26.,0.005,0.,'x')))
json.dump(out,open('Uall.json','w'))
