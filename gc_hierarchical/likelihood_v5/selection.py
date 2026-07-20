
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

class SelectionModel:
    def __init__(self,cfg): self.cfg=cfg

    @staticmethod
    def smooth_ratio(num,den,sigma,alpha=1.0,beta=1.0):
        n=gaussian_filter(num.astype(float),sigma=sigma,mode="nearest")
        d=gaussian_filter(den.astype(float),sigma=sigma,mode="nearest")
        return (n+alpha)/(d+alpha+beta)

    def fit(self,k):
        R=pd.to_numeric(k["Rad_arcmin"],errors="coerce").to_numpy()
        entry=k["in_candidate_catalog"].astype(bool).to_numpy()
        selected=k["selected_gold_silver_085"].astype(bool).to_numpy()
        r=pd.to_numeric(k["rmag"],errors="coerce").to_numpy()

        self.entry_edges=np.array([10,20,35,55,80,110,150,220],float)
        ok=np.isfinite(R)&(R>=10)
        den,_=np.histogram(R[ok],self.entry_edges)
        num,_=np.histogram(R[ok&entry],self.entry_edges)
        ec=.5*(self.entry_edges[:-1]+self.entry_edges[1:])
        ef=self.smooth_ratio(num,den,.8)
        self.entry=RegularGridInterpolator((ec,),ef,bounds_error=False,fill_value=None)

        self.r_edges=np.array([16.5,18.0,18.8,19.5,20.2,20.8,21.3,21.8,22.5])
        self.R_edges=np.array([10,25,50,80,115,160,220])
        usable=ok&entry&np.isfinite(r)
        den2,_,_=np.histogram2d(r[usable],R[usable],bins=[self.r_edges,self.R_edges])
        num2,_,_=np.histogram2d(r[usable&selected],R[usable&selected],bins=[self.r_edges,self.R_edges])
        rc=.5*(self.r_edges[:-1]+self.r_edges[1:])
        Rc=.5*(self.R_edges[:-1]+self.R_edges[1:])
        rf=self.smooth_ratio(num2,den2,(1.0,.8))
        self.rank=RegularGridInterpolator((rc,Rc),rf,bounds_error=False,fill_value=None)
        return self

    def probability(self,r,R):
        r=np.asarray(r,float); R=np.asarray(R,float)
        pe=self.entry(np.clip(R,self.entry_edges[0],self.entry_edges[-1])[:,None]).ravel()
        pts=np.column_stack([
          np.clip(r,self.r_edges[0],self.r_edges[-1]),
          np.clip(R,self.R_edges[0],self.R_edges[-1])
        ])
        pr=self.rank(pts).ravel()
        return np.clip(pe*pr,1e-5,.999)
