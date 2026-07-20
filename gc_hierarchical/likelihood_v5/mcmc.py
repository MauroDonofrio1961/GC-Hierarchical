
import json, numpy as np, pandas as pd

def run_mcmc(cfg,posterior,map_theta,out):
    import emcee
    nwalk=cfg["mcmc"]["walkers"]; ndim=len(map_theta)
    if nwalk<2*ndim: nwalk=2*ndim+4
    rng=np.random.default_rng(cfg["project"]["seed"])
    bounds=np.asarray(cfg["parameters"]["bounds"],float)
    scale=cfg["mcmc"]["initial_fractional_spread"]*(bounds[:,1]-bounds[:,0])
    p0=map_theta[None,:]+rng.normal(size=(nwalk,ndim))*scale[None,:]
    p0=np.clip(p0,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
    sampler=emcee.EnsembleSampler(nwalk,ndim,posterior)
    sampler.run_mcmc(p0,cfg["mcmc"]["burn_steps"],progress=True)
    sampler.reset()
    sampler.run_mcmc(None,cfg["mcmc"]["production_steps"],progress=True)
    chain=sampler.get_chain(flat=True,thin=cfg["mcmc"]["thin"])
    logp=sampler.get_log_prob(flat=True,thin=cfg["mcmc"]["thin"])
    names=cfg["parameters"]["names"]
    df=pd.DataFrame(chain,columns=names); df["log_posterior"]=logp
    df.to_csv(out/"posterior_samples.csv",index=False)
    q={}
    for i,name in enumerate(names):
        p16,p50,p84=np.percentile(chain[:,i],[16,50,84])
        q[name]={"p16":float(p16),"median":float(p50),"p84":float(p84)}
    (out/"posterior_summary.json").write_text(json.dumps(q,indent=2))
    return chain,q
