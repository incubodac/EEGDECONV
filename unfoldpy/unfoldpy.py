from sklearn.base import BaseEstimator
import nupy as np





class Unfolder(BaseEstimator):
     def __init__(
        self,
        tmin,
        tmax,
        sfreq,
        feature_names=None,
        estimator=None,
        fit_intercept=None,
        scoring="r2",
        n_jobs=None,
        verbose=None,
    ):
        self.feature_names = feature_names
        self.sfreq = float(sfreq)
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = 0.0 if estimator is None else estimator
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.patterns = patterns
        self.n_jobs = n_jobs
        self.edge_correction = edge_correction
        
    def __repr__(self):  # 
        s = "tmin, tmax : (%.3f, %.3f), " % (self.tmin, self.tmax)
        estimator = self.estimator
        if not isinstance(estimator, str):
            estimator = type(self.estimator)
        s += "estimator : %s, " % (estimator,)
        if hasattr(self, "coef_"):
            if self.feature_names is not None:
                feats = self.feature_names
                if len(feats) == 1:
                    s += "feature: %s, " % feats[0]
                else:
                    s += "features : [%s, ..., %s], " % (feats[0], feats[-1])
            s += "fit: True"
        else:
            s += "fit: False"
        if hasattr(self, "scores_"):
            s += "scored (%s)" % self.scoring
        return "<Unfolder | %s>" % s
        
        
    
    def fit(self, X, y):
        # Implement the fitting logic for your estimator
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # Your fitting code here
        # Example: self.model = SomeModel()
        #          self.model.fit(X, y)
        return self
    
    def predict(self, X):
        # Implement the prediction logic for your estimator
        # X: array-like, shape (n_samples, n_features)
        # Your prediction code here
        # Example: return self.model.predict(X)
        pass
    
    def score(self, X, y):
        # Implement the scoring logic for your estimator
        # X: array-like, shape (n_samples, n_features)
        # y: array-like, shape (n_samples,)
        # Your scoring code here
        # Example: return self.model.score(X, y)
        pass
    
def _times_to_samples(tmin, tmax, sfreq):
    """Convert a tmin/tmax in seconds to samples."""
    # Convert seconds to samples
    smp_ix = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))
    return smp_ix
