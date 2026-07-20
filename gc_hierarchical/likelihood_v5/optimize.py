
import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

def optimize_map(cfg,posterior,mode,out):
    bounds=np.asarray(cfg["parameters"]["bounds"],float)
    n=cfg["optimizer"]["science_random_points"] if mode=="science" else cfg["optimizer"]["quick_random_points"]
    X=qmc.scale(
      qmc.LatinHypercube(d=len(bounds),seed=cfg["project"]["seed"]).random(n),
      bounds[:,0],bounds[:,1])
    logp=np.array([posterior(x) for x in X])
    order=np.argsort(logp)[::-1]
    hist=[]
    for i in range(n):
        hist.append({"stage":"latin","log_posterior":float(logp[i]),
                     **dict(zip(cfg["parameters"]["names"],X[i]))})
    bestx=X[order[0]].copy(); bestlp=logp[order[0]]
    for j in order[:cfg["optimizer"]["local_starts"]]:
        res=minimize(lambda t:-posterior(t),X[j],method="Powell",bounds=bounds,
          options={"maxiter":cfg["optimizer"]["local_maxiter"],"xtol":2e-3,"ftol":2e-3})
        lp=-res.fun
        hist.append({"stage":"local","log_posterior":float(lp),
                     **dict(zip(cfg["parameters"]["names"],res.x))})
        if lp>bestlp: bestlp,bestx=lp,res.x.copy()
    pd.DataFrame(hist).sort_values("log_posterior",ascending=False).to_csv(
      out/"optimization_history.csv",index=False)
    result={"mode":mode,"log_posterior":float(bestlp),"calls":posterior.calls,
            "parameters":{k:float(v) for k,v in zip(cfg["parameters"]["names"],bestx)}}
    (out/"map_parameters.json").write_text(json.dumps(result,indent=2))
    return bestx,result
