
from pathlib import Path
import json, numpy as np

def load_config(path):
    p=Path(path).resolve()
    cfg=json.loads(p.read_text())
    cfg["_base"]=str(p.parent)
    return cfg

def resolve(cfg, value):
    p=Path(value)
    return p if p.is_absolute() else Path(cfg["_base"])/p

def ensure(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p

def angular_radius_arcmin(ra,dec,ra0,dec0):
    ra,dec,ra0,dec0=map(np.deg2rad,[ra,dec,ra0,dec0])
    c=np.sin(dec)*np.sin(dec0)+np.cos(dec)*np.cos(dec0)*np.cos(ra-ra0)
    return np.rad2deg(np.arccos(np.clip(c,-1,1)))*60.0

def logsumexp3(a,b,c):
    m=np.maximum(np.maximum(a,b),c)
    return m+np.log(np.exp(a-m)+np.exp(b-m)+np.exp(c-m))
