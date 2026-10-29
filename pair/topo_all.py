import json,glob,os,numpy as np
from topo import analyse
J=lambda t: json.load(open(t+'.json'))['E']
rows=[]
def add(f,d,mode,fam,branch,lam):
    if not os.path.exists(f+'.npz'): return
    cores={'pair':[(-d/2,0.),(d/2,0.)],'R':[(d/2,0.)],'none':[]}[mode]
    o,_=analyse(f,cores); o.update(file=f,d=d,mode=mode,fam=fam,branch=branch,lam=lam,E=J(f)); rows.append(o)
    c=o['cores']; print(f'{f:42s} d={d:4.0f} E={o["E"]:8.3f} N+={o["Nplus"]:2d} N-={o["Nminus"]:2d} Qbox={o["Qbox"]} gauss={o["gauss"]} | '+'  '.join(f'Q={p["Qhole"]} N-loc={p["Nminus_local"]} z7={p["z7"]}' for p in c),flush=True)
L='_lam-0.5'
add('none_d0.0_V0.005'+L,0,'none','twist','bg',-0.5); add('none_d0.0_V0.005_a0_x'+L,0,'none','mismatch','bg',-0.5)
add('none_d0.0_V0.005',0,'none','twist','bg',1.0)
ds=[8.,16.,21.,26.,30.,34.,40.,48.,56.]
for d in ds:
    add(f'R_d{d}_V0.005'+L,d,'R','twist','single',-0.5)
    add(f'pair_d{d}_V0.005'+L,d,'pair','twist','default',-0.5)
    for f in glob.glob(f'pair_d{d}_V0.005{L}_from_*.npz'): add(f[:-4],d,'pair','twist','seed',-0.5)
    add(f'swout_d{d}',d,'pair','twist','pull',-0.5); add(f'swin_d{d}',d,'pair','twist','push',-0.5)
    add(f'swBout_d{d}',d,'pair','twist','pull',-0.5)
    add(f'twL1out_d{d}',d,'pair','twist','pull',1.0); add(f'twL1in_d{d}',d,'pair','twist','push',1.0); add(f'twL1Bout_d{d}',d,'pair','twist','pull',1.0)
    add(f'R_d{d}_V0.005_a0_x'+L,d,'R','mismatch','single',-0.5)
    add(f'pair_d{d}_V0.005_a0_x'+L,d,'pair','mismatch','default',-0.5)
    add(f'mmout_d{d}',d,'pair','mismatch','pull',-0.5); add(f'mmin_d{d}',d,'pair','mismatch','push',-0.5)
for d in [42.,44.,46.]: add(f'swBout_d{d}',d,'pair','twist','pull',-0.5)
json.dump(rows,open('topo_all.json','w'))
