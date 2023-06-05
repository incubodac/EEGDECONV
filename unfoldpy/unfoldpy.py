from sklearn.base import BaseEstimator
import numpy as np





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
        self.n_jobs = n_jobs
        
    def __repr__(self):  
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
        """Fit a receptive field model.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_features)
            The input features for the model.
        y : array, shape (n_times[, n_epochs][, n_outputs])
            The output features for the model.

        Returns
        -------
        self : instance
            The instance so you can chain operations.
        """
        from scipy import linalg

        if self.scoring not in _SCORERS.keys():
            raise ValueError(
                "scoring must be one of %s, got"
                "%s " % (sorted(_SCORERS.keys()), self.scoring)
            )
        from sklearn.base import clone

        X, y, _, self._y_dim = self._check_dimensions(X, y)

        if self.tmin > self.tmax:
            raise ValueError(
                "tmin (%s) must be at most tmax (%s)" % (self.tmin, self.tmax)
            )
        # Initialize delays
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # Define the slice that we should use in the middle
        self.valid_samples_ = _delays_to_slice(self.delays_)

        if isinstance(self.estimator, numbers.Real):
            if self.fit_intercept is None:
                self.fit_intercept = True
            estimator = TimeDelayingRidge(
                self.tmin,
                self.tmax,
                self.sfreq,
                alpha=self.estimator,
                fit_intercept=self.fit_intercept,
                n_jobs=self.n_jobs,
                edge_correction=self.edge_correction,
            )
        elif is_regressor(self.estimator):
            estimator = clone(self.estimator)
            if (
                self.fit_intercept is not None
                and estimator.fit_intercept != self.fit_intercept
            ):
                raise ValueError(
                    "Estimator fit_intercept (%s) != initialization "
                    "fit_intercept (%s), initialize ReceptiveField with the "
                    "same fit_intercept value or use fit_intercept=None"
                    % (estimator.fit_intercept, self.fit_intercept)
                )
            self.fit_intercept = estimator.fit_intercept
        else:
            raise ValueError(
                "`estimator` must be a float or an instance"
                " of `BaseEstimator`,"
                " got type %s." % type(self.estimator)
            )
        self.estimator_ = estimator
        del estimator
        _check_estimator(self.estimator_)

        # Create input features
        n_times, n_epochs, n_feats = X.shape
        n_outputs = y.shape[-1]
        n_delays = len(self.delays_)

        # Update feature names if we have none
        if (self.feature_names is not None) and (len(self.feature_names) != n_feats):
            raise ValueError(
                "n_features in X does not match feature names "
                "(%s != %s)" % (n_feats, len(self.feature_names))
            )

        # Create input features
        X, y = self._delay_and_reshape(X, y)

        self.estimator_.fit(X, y)
        coef = get_coef(self.estimator_, "coef_")  # (n_targets, n_features)
        shape = [n_feats, n_delays]
        if self._y_dim > 1:
            shape.insert(0, -1)
        self.coef_ = coef.reshape(shape)

        # Inverse-transform model weights
        if self.patterns:
            if isinstance(self.estimator_, TimeDelayingRidge):
                cov_ = self.estimator_.cov_ / float(n_times * n_epochs - 1)
                y = y.reshape(-1, y.shape[-1], order="F")
            else:
                X = X - X.mean(0, keepdims=True)
                cov_ = np.cov(X.T)
            del X

            # Inverse output covariance
            if y.ndim == 2 and y.shape[1] != 1:
                y = y - y.mean(0, keepdims=True)
                inv_Y = linalg.pinv(np.cov(y.T))
            else:
                inv_Y = 1.0 / float(n_times * n_epochs - 1)
            del y

            # Inverse coef according to Haufe's method
            # patterns has shape (n_feats * n_delays, n_outputs)
            coef = np.reshape(self.coef_, (n_feats * n_delays, n_outputs))
            patterns = cov_.dot(coef.dot(inv_Y))
            self.patterns_ = patterns.reshape(shape)

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




# def _times_to_delays(tmin, tmax, sfreq):
#     """Convert a tmin/tmax in seconds to delays."""
#     # Convert seconds to samples
#     delays = np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq) + 1))
#     return delays


# def _delays_to_slice(delays):
#     """Find the slice to be taken in order to remove missing values."""
#     # Negative values == cut off rows at the end
#     min_delay = None if delays[-1] <= 0 else delays[-1]
#     # Positive values == cut off rows at the end
#     max_delay = None if delays[0] >= 0 else delays[0]
#     return slice(min_delay, max_delay)



def _r2_score(y_true, y, multioutput=None):
    from sklearn.metrics import r2_score

    return r2_score(y_true, y, multioutput=multioutput)


_SCORERS = {"r2": _r2_score}#, "corrcoef": _corr_score}

if __name__=='__main__':
    from sklearn.linear_model import LinearRegression
    feature_cols = ['mss']
    intercept    = ['fixation']
    tmin ,tmax = -.2 , .4
    sfreq = 500
    unf=Unfolder(
         tmin, tmax, sfreq, feature_cols, estimator=LinearRegression(),scoring='r2'
    )
    print(unf)