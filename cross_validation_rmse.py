from get_data import get_data
from model_pytorch import run_model
import numpy as np
import pandas as pd
import numpy.ma as ma
import pickle
from sklearn.model_selection import KFold

class OurRobustToNanScaler():
    """
    This class is equal to StandardScaler from sklearn but can work with NaN's (ignoring it) but
    sklearn's scaler can't do it.
    """

    def fit(self, data):
        masked = ma.masked_invalid(data)
        self.means = np.mean(masked, axis=0)
        self.stds = np.std(masked, axis=0)

    def fit_transform(self, data):
        self.fit(data)
        masked = ma.masked_invalid(data)
        masked -= self.means
        masked /= self.stds
        return ma.getdata(masked)

    def inverse_transform(self, data):
        masked = ma.masked_invalid(data)
        masked *= self.stds
        masked += self.means
        return ma.getdata(masked)

def cross_validation(ya_file_name):
    df, smiles2descriptors = get_data(ya_file_name)
    #5-fold cross-validation
    x = np.array(list(smiles2descriptors.values()))
    y = df.values

    input_scaler = OurRobustToNanScaler()
    output_scaler = OurRobustToNanScaler()
    x = input_scaler.fit_transform(x)
    y = output_scaler.fit_transform(y)
    mse_cv = []
    r2_cv = []
    with open("scaler.pickle", 'wb') as f:
        pickle.dump(output_scaler, f)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mse_vals, r2_vals = run_model(X_train, X_test, y_train, y_test,cuda=True,log_file_name=None,
                                      log_dir_name=("runs/model1826_512_cv_fold"+str(fold)), number_of_epochs=2000,
                                      ckpt_name=("model1826_512_cv_fold"+str(fold)+".pt"), scaler=output_scaler)
        fold+=1
        mse_cv.append(mse_vals)
        r2_cv.append(r2_vals)
    print("Calculation finished")
    mse_cv = np.array(mse_cv)
    r2_cv = np.array(r2_cv)
    mse_cv = np.mean(mse_cv,axis = 0)
    r2_cv = np.mean(r2_cv, axis = 0)
    print(mse_cv)
    print(r2_cv)
    return list(df.columns), mse_cv, r2_cv
if __name__ == '__main__':
    endpoints, mse_cv, r2_cv = cross_validation("config_endpoints.yml")
    with open("output_cv.txt", 'w') as f:
        f.write("Endpoint MSE     r2      \n")
        for idx, endpoint in enumerate(endpoints):
            f.write(endpoint+" "+str(mse_cv[idx])+" "+str(r2_cv[idx])+"\n")