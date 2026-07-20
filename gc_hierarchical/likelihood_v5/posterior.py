
import numpy as np
from scipy.special import gammaln
from .utils import logsumexp3

class Posterior:
    def __init__(self,cfg,science,gc,bg):
        self.cfg,self.data,self.gc,self.bg=cfg,science,gc,bg
        self.names=cfg["parameters"]["names"]
        self.bounds=np.asarray(cfg["parameters"]["bounds"],float)
        self.lbg=bg.score(science)
        self.rank=science.Rank.str.lower().to_numpy()
        self.calls=0

    def inside(self,t):
        return np.all(t>=self.bounds[:,0]) and np.all(t<=self.bounds[:,1])

    @staticmethod
    def softmax3(blue_logit, red_logit):
        logits=np.array([blue_logit,red_logit,0.0],dtype=float)
        logits-=np.max(logits)
        p=np.exp(logits)
        p/=p.sum()
        return p

    def rank_fractions(self,t):
        d=dict(zip(self.names,t))
        gold=self.softmax3(d["gold_blue_logit"],d["gold_red_logit"])
        silver=self.softmax3(d["silver_blue_logit"],d["silver_red_logit"])
        return gold,silver

    @staticmethod
    def dirichlet_logpdf(p, mean, concentration):
        p=np.clip(np.asarray(p,float),1e-12,1.0)
        alpha=np.asarray(mean,float)*float(concentration)
        return gammaln(alpha.sum())-gammaln(alpha).sum()+np.sum((alpha-1.0)*np.log(p))

    def logprior(self,t):
        if not self.inside(t):
            return -np.inf
        d=dict(zip(self.names,t))
        lp=0.0

        for name in ["logM0_blue","logM0_red","sigma_blue","sigma_red",
                     "blue_logz_mean","red_logz_mean","blue_logz_sigma",
                     "red_logz_sigma","delta_g","delta_r",
                     "extra_color_scatter","blue_radius_scale","red_radius_scale"]:
            mu,sd=self.cfg["priors"][name]
            lp-=0.5*((d[name]-mu)/sd)**2

        if d["blue_logz_mean"]>=d["red_logz_mean"]:
            lp-=self.cfg["priors"]["ordering_penalty"]*(
                d["blue_logz_mean"]-d["red_logz_mean"]+0.03
            )**2

        gold,silver=self.rank_fractions(t)
        gp=self.cfg["priors"]["gold_fraction_prior"]
        sp=self.cfg["priors"]["silver_fraction_prior"]
        lp+=self.dirichlet_logpdf(gold,gp["mean"],gp["concentration"])
        lp+=self.dirichlet_logpdf(silver,sp["mean"],sp["concentration"])

        minbg=self.cfg["priors"]["minimum_background_fraction"]
        maxbg=self.cfg["priors"]["maximum_background_fraction"]
        for p in (gold,silver):
            if p[2] < minbg:
                lp-=5000.0*(minbg-p[2])**2
            if p[2] > maxbg:
                lp-=5000.0*(p[2]-maxbg)**2
        return lp

    def fractions(self,t):
        gold,silver=self.rank_fractions(t)
        blue=np.where(self.rank=="gold",gold[0],silver[0])
        red=np.where(self.rank=="gold",gold[1],silver[1])
        bg=np.where(self.rank=="gold",gold[2],silver[2])

        # hard numerical validation
        s=blue+red+bg
        if not np.allclose(s,1.0,rtol=0,atol=1e-12):
            raise RuntimeError("Mixture fractions do not sum to one.")
        if np.any(blue<=0) or np.any(red<=0) or np.any(bg<=0):
            raise RuntimeError("Mixture fractions must be strictly positive.")
        return blue,red,bg

    def __call__(self,t,return_membership=False):
        self.calls+=1
        t=np.asarray(t,float)
        lp=self.logprior(t)
        if not np.isfinite(lp):
            return -np.inf if not return_membership else (-np.inf,None)

        lb,lr=self.gc.logdensities(t)
        fb,fr,fg=self.fractions(t)
        a=np.log(fb)+lb
        b=np.log(fr)+lr
        c=np.log(fg)+self.lbg
        ll=logsumexp3(a,b,c)
        value=lp+ll.sum()

        if return_membership:
            pb=np.exp(a-ll)
            pr=np.exp(b-ll)
            pg=np.exp(c-ll)
            membership=np.column_stack([pb,pr,pg])

            rowsum=membership.sum(axis=1)
            if not np.all(np.isfinite(membership)):
                raise RuntimeError("Non-finite posterior membership probability.")
            if not np.allclose(rowsum,1.0,rtol=1e-10,atol=1e-10):
                raise RuntimeError("Posterior memberships do not sum to one.")
            if np.any(membership<0) or np.any(membership>1):
                raise RuntimeError("Posterior memberships outside [0,1].")
            return value,membership
        return value
