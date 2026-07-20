
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

class BackgroundDensity:
    """
    GMM background in (g, r, log R). NumPy arrays are used consistently
    to avoid sklearn feature-name warnings.
    """
    def __init__(self,cfg,science,background):
        self.cols=["gmag","rmag","log_radius"]
        xs=science[self.cols].to_numpy(dtype=float)
        xb=background[self.cols].to_numpy(dtype=float)
        combined=np.vstack([xs,xb])
        self.scaler=StandardScaler().fit(combined)
        zbg=self.scaler.transform(xb)
        self.gmm=GaussianMixture(
          n_components=cfg["background_control"]["gmm_components"],
          covariance_type="full",
          reg_covar=cfg["background_control"]["regularization"],
          random_state=cfg["project"]["seed"],
          n_init=3
        ).fit(zbg)

    def score(self,d):
        x=d[self.cols].to_numpy(dtype=float)
        z=self.scaler.transform(x)
        return self.gmm.score_samples(z)
