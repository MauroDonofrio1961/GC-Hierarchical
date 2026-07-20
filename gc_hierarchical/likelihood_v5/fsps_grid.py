
import numpy as np
from scipy.interpolate import interp1d

class FSPSGrid:
    def __init__(self,cfg):
        import fsps
        s=cfg["stellar_population"]
        lo,hi,n=s["fsps_logz_grid"]
        self.logz=np.linspace(lo,hi,int(n))
        self.bands=s["filters"]
        sp=fsps.StellarPopulation(
          zcontinuous=1,sfh=0,imf_type=s["imf_type"],
          compute_vega_mags=False,dust_type=0,add_dust_emission=False)
        mags=[]
        for z in self.logz:
            sp.params["logzsol"]=float(z)
            mags.append(sp.get_mags(tage=s["age_gyr"],bands=self.bands))
        mags=np.asarray(mags,float)
        self.interp=[
          interp1d(self.logz,mags[:,j],kind="cubic",
                   bounds_error=False,fill_value="extrapolate")
          for j in range(2)
        ]

    def absolute_mags_per_unit_mass(self,logz):
        z=np.asarray(logz,float)
        return np.column_stack([f(z) for f in self.interp])
