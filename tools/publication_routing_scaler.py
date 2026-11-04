"""Stable serialization identity for the study's origin-weighted scaler."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class OriginScaler(StandardScaler):
    """Translate before moment accumulation so constant features have zero variance."""
    def fit(self,x,y=None,sample_weight=None):
        values=np.asarray(x,dtype=float)
        offset=values[0].copy()
        super().fit(values-offset,y,sample_weight=sample_weight)
        self.mean_+=offset
        return self

