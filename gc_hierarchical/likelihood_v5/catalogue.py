
import numpy as np, pandas as pd
from .utils import resolve, angular_radius_arcmin

def load_catalogues(cfg):
    c=pd.read_csv(resolve(cfg,cfg["data"]["candidate_catalogue"]))
    k=pd.read_csv(resolve(cfg,cfg["data"]["confirmed_catalogue"]))
    c["Rank"]=c["Rank"].astype(str).str.lower().str.strip()
    for x in ["RAdeg","DEdeg","gmag","rmag","e_gmag","e_rmag","TotL"]:
        c[x]=pd.to_numeric(c[x],errors="coerce")
    c["color"]=c["gmag"]-c["rmag"]
    c["radius_arcmin"]=angular_radius_arcmin(
        c["RAdeg"],c["DEdeg"],
        cfg["galaxy"]["ra_center_deg"],cfg["galaxy"]["dec_center_deg"])
    c["log_radius"]=np.log10(c["radius_arcmin"])
    return c,k

def observable_window(d,cfg):
    s=cfg["sample"]
    return (
      d.gmag.between(*s["g_range"]) &
      d.rmag.between(*s["r_range"]) &
      d.color.between(*s["color_range"]) &
      d.radius_arcmin.between(*s["radius_range_arcmin"])
    )

def science_sample(c,cfg):
    s=cfg["sample"]
    q=c.Rank.isin(s["ranks"]) & (c.TotL>=s["minimum_total_likelihood"]) & observable_window(c,cfg)
    return c.loc[q].dropna(subset=["gmag","rmag","color","radius_arcmin"]).reset_index(drop=True)

def background_sample(c,cfg):
    q=c.Rank.isin(cfg["background_control"]["ranks"]) & observable_window(c,cfg)
    d=c.loc[q].dropna(subset=["gmag","rmag","color","radius_arcmin"]).copy()
    n=cfg["background_control"]["maximum_objects"]
    if len(d)>n:
        d=d.sample(n=n,random_state=cfg["project"]["seed"])
    return d.reset_index(drop=True)
