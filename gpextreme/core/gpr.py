import numpy as np
import sklearn.gaussian_process as skgp
from scipy.linalg import cho_solve, cholesky
from sklearn.preprocessing._data import _handle_zeros_in_scale

class GaussianProcessRegressor(skgp.GaussianProcessRegressor):
    
    def predict_grids_fast(self, x, y, grids, K_cache, mean, var): 
        '''
        after adding one sample, predicting the new mean and variance. 
        x is the newly added point
        '''
        n_D = len(self.X_train_)
        x = x.reshape(1,-1)
        k_x_grid = self.kernel_(x, grids)
        post_mean, post_std = self.predict(x, return_std=True)
        post_cov = (k_x_grid.T - K_cache[:n_D].T @ cho_solve((self.L_, True), 
                                                self.kernel(self.X_train_, x),
                                                check_finite=False)).reshape(-1)
        mean += (post_cov/ post_std ** 2) * (y - post_mean)
        var -= (post_cov / post_std) ** 2 
        K_cache[n_D:n_D+1] = k_x_grid
        return mean, var
    
    
    def fit_keep(self, X, y):
        '''Same GPR with different dataset.
        '''
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), 
                                                       copy=False)
            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1 
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self