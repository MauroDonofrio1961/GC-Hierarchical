
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

class ComponentDensity:
    def __init__(self,cfg,science,selection,fsps,zgrid):
        self.cfg,self.data,self.selection,self.fsps=cfg,science,selection,fsps
        lg=cfg["latent_grid"]
        self.logm=np.linspace(*lg["logmass"][:2],int(lg["logmass"][2]))
        self.zgrid=zgrid
        self.Mssp=fsps.absolute_mags_per_unit_mass(zgrid)
        self.dm=cfg["galaxy"]["distance_modulus"]
        e=cfg["foreground_extinction"]
        self.Ag=e["ebv"]*e["A_over_EBV_g"]
        self.Ar=e["ebv"]*e["A_over_EBV_r"]
        self.chunk=int(lg["chunk_size"])
        self.g=science.gmag.to_numpy()
        self.r=science.rmag.to_numpy()
        self.R=science.radius_arcmin.to_numpy()
        self.eg=np.clip(science.e_gmag.fillna(.04).to_numpy(),.008,.35)
        self.er=np.clip(science.e_rmag.fillna(.03).to_numpy(),.008,.35)

    @staticmethod
    def logweights(grid,mu,sigma):
        w=norm.pdf(grid,loc=mu,scale=sigma)
        w=np.clip(w,1e-300,None); w/=w.sum()
        return np.log(w)

    @staticmethod
    def radial_logpdf(R,scale):
        # projected exponential-like profile, normalized on the fitted radial interval
        R=np.asarray(R,float)
        raw=np.clip(R,1e-6,None)*np.exp(-R/scale)
        return np.log(np.clip(raw,1e-300,None))

    def logdensity(self,logM0,sigmaM,zmean,zsigma,dg,dr,xcs,rscale):
        logwm=self.logweights(self.logm,logM0,sigmaM)
        logwz=self.logweights(self.zgrid,zmean,zsigma)

        gpred=(self.Mssp[:,0][None,:]-2.5*self.logm[:,None])+self.dm+self.Ag+dg
        rpred=(self.Mssp[:,1][None,:]-2.5*self.logm[:,None])+self.dm+self.Ar+dr
        gp, rp = gpred.ravel(), rpred.ravel()
        latent=(logwm[:,None]+logwz[None,:]).ravel()

        out=np.empty(len(self.data))
        for start in range(0,len(self.data),self.chunk):
            stop=min(start+self.chunk,len(self.data))
            go=self.g[start:stop,None]; ro=self.r[start:stop,None]
            sg=np.sqrt(self.eg[start:stop,None]**2+0.5*xcs**2)
            sr=np.sqrt(self.er[start:stop,None]**2+0.5*xcs**2)
            llg=-0.5*((go-gp[None,:])/sg)**2-np.log(sg)-0.5*np.log(2*np.pi)
            llr=-0.5*((ro-rp[None,:])/sr)**2-np.log(sr)-0.5*np.log(2*np.pi)

            Robj=self.R[start:stop]
            # use each object's radius in the selection term
            block=np.empty((stop-start,len(rp)))
            for j,Rj in enumerate(Robj):
                block[j]=np.log(self.selection.probability(rp,np.full(len(rp),Rj)))
            out[start:stop]=logsumexp(llg+llr+latent[None,:]+block,axis=1)

        # add radial population term
        out += self.radial_logpdf(self.R,rscale)
        return out

class ThreeComponentDensity:
    def __init__(self,cfg,science,selection,fsps):
        lg=cfg["latent_grid"]
        zb=np.linspace(*lg["logz_blue"][:2],int(lg["logz_blue"][2]))
        zr=np.linspace(*lg["logz_red"][:2],int(lg["logz_red"][2]))
        self.blue=ComponentDensity(cfg,science,selection,fsps,zb)
        self.red=ComponentDensity(cfg,science,selection,fsps,zr)

    def logdensities(self,t):
        (mub,sb,mur,sr,zbm,zbs,zrm,zrs,
         gold_blue_logit,gold_red_logit,
         silver_blue_logit,silver_red_logit,
         dg,dr,xcs,Rb,Rr)=t
        lb=self.blue.logdensity(mub,sb,zbm,zbs,dg,dr,xcs,Rb)
        lr=self.red.logdensity(mur,sr,zrm,zrs,dg,dr,xcs,Rr)
        return lb,lr
