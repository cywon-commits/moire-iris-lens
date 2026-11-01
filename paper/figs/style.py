import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, sys, os
sys.path.insert(0,'/home/claude/ridge/hex'); sys.path.insert(0,'/home/claude/ridge/pair'); sys.path.insert(0,'/home/claude/ridge/big')
P='/home/claude/ridge/pair/'; B='/home/claude/ridge/big/'; Dd='/home/claude/paper/data/'
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':6.8,'xtick.labelsize':7,'ytick.labelsize':7,
  'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'dejavuserif','axes.linewidth':0.6,'lines.linewidth':1.1,
  'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,
  'legend.handlelength':1.6,'legend.borderaxespad':0.3,'savefig.dpi':300})
W1=3.375; W2=7.0
CT='#c0392b'; CM='#2471a3'; CG='#7f8c8d'
s3=np.sqrt(3); TH=1/30.; AC=s3/2; RL=np.sqrt(120.)
def plabel(ax,s,title='',**k): ax.set_title(r'$\bf{'+s.strip('()')+'}$ '+title if False else s+' '+title,loc='left',pad=3,fontsize=8)
def save(fig,name):
    fig.savefig(name+'.pdf'); fig.savefig(name+'_preview.png',dpi=170)
def Vmap(D):
    from hexcore import Vhex
    V,_,_=Vhex(D['ux'],D['uy'],0.005,s3*0.005); V=V.copy(); V[D['core']]=np.nan; return V
