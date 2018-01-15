# import synapseclient
import pandas as pd
from sklearn import svm
import os
import numpy as np
from JSON_Parameters import Parameter

from multiprocessing import Process, Manager


def fit(model, train_df, synergy_score):
    model.fit(train_df, synergy_score)
    return model


def make_svr_model(train_df, synergy_score, C = 120, gamma=0.081, epsilon=0.1, poly=False):
    if poly == True:
        model = svm.SVR(kernel='poly', C=C, degree=gamma)
    else:
        model = svm.SVR(kernel='rbf', C=C)
    model = fit(model, train_df, synergy_score)
    return model

def make_svc_model(train_df, synergy_score, kernel, C = 120, degree=3):
    model = svm.SVC(C=C, kernel=str(kernel), degree=degree)
    model = fit(model, train_df, synergy_score)
    return model

def make_dtc_model(train_df, synergy_score):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(min_samples_split  = 5, min_samples_leaf=10)
    model = fit(model, train_df, synergy_score)
    return model
def make_rfc_model(train_df, synergy_score, n_estimators , max_features ):
    from sklearn import ensemble
    model = ensemble.RandomForestClassifier(n_estimators =n_estimators , max_features =max_features, n_jobs=-1)
    model = fit(model, train_df, synergy_score)
    return model
def make_etc_model(train_df, synergy_score, n_estimators, max_features):
    from sklearn import ensemble
    model = ensemble.ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1)
    model = fit(model, train_df, synergy_score)
    return model
def make_lrc_model(train_df, synergy_score,C,solver):
    from sklearn import linear_model
    model = linear_model.LogisticRegression(C=C, solver=solver,max_iter=5000)
    model = fit(model, train_df, synergy_score)
    return model

def make_rf_model(train_df, synergy_score,n_tree = 120, max_features = 0.33):
    from sklearn import ensemble
    model = ensemble.RandomForestRegressor(n_estimators=n_tree, max_features = max_features, bootstrap=True, n_jobs=-1)

    model = fit(model, train_df, synergy_score)

    return model

def make_extra_model(train_df, synergy_score, n_tree = 120, max_features = 0.33):
    from sklearn import ensemble
    model = ensemble.ExtraTreesRegressor(n_estimators=n_tree, max_features = max_features, bootstrap=True, n_jobs=-1)
    model = fit(model, train_df, synergy_score)

    return model

def make_ada_model(train_df, synergy_score, n_tree = 120, base_estimator="RF"):
    from sklearn import ensemble
    from sklearn import kernel_ridge
    if base_estimator=="RF":
        base_estimator_ = ensemble.RandomForestRegressor()
    elif base_estimator =="Kernel":
        base_estimator_ = kernel_ridge.KernelRidge(alpha=0.01, kernel='poly', degree=2)

    model = ensemble.AdaBoostRegressor(n_estimators=n_tree, base_estimator = base_estimator_, learning_rate=0.9)
    model = fit(model, train_df, synergy_score)

    return model

def make_rf_classification_model(train_df, synergy_score,n_tree = 120, max_features = 0.33, min_samples_leaf=10):
    from sklearn import ensemble
    model = ensemble.RandomForestClassifier(n_estimators=n_tree, max_features = max_features, bootstrap=True, n_jobs=-1, min_samples_leaf=10)
    model = fit(model, train_df, synergy_score)

    return model


def make_ridge_model(train_df, synergy_score, alpha=1.0):
    from sklearn import linear_model
    model = linear_model.Ridge(alpha=alpha)
    model = fit(model, train_df, synergy_score)
    return model

def make_lasso_model(train_df, synergy_score, alpha=1.0):
    from sklearn import linear_model
    model = linear_model.Lasso(alpha=alpha, max_iter=3000)
    model = fit(model, train_df, synergy_score)
    return model

def make_elasticnet_model(train_df, synergy_score, alpha=1.0, l1_ratio=0.1):
    from sklearn import linear_model
    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model = fit(model, train_df, synergy_score)
    return model

def make_kernelridge_model(train_df, synergy_score, alpha=1.0):
    from sklearn import kernel_ridge
    model = kernel_ridge.KernelRidge(alpha=alpha, kernel='poly', degree=2)
    model = fit(model, train_df, synergy_score)
    return model

def make_sgd_model(train_df, synergy_score, alpha=1.0):
    from sklearn import linear_model
    model = linear_model.SGDRegressor(alpha=alpha, learning_rate='optimal')
    model = fit(model, train_df, synergy_score)
    return model

def make_nn_model(train_df, synergy_score, nlayer=3):
    from sklearn import neural_network
    if nlayer == 1:
        hlayer = (100,)
    elif nlayer == 2:
        hlayer = (100,100)
    elif nlayer == 3:
        hlayer = (100,100,100)
    elif nlayer == 4:
        hlayer = (100,100,100,100)
    print hlayer
    model = neural_network.MLPRegressor(hidden_layer_sizes=hlayer, learning_rate='constant')
    model = fit(model, train_df, synergy_score)
    return model

def realpredict(model, test_df, index, indexname, fname, repeat = 2):
    tmp = list()
    return_dict = dict()
    for k in range(repeat):
        X_t = test_df.copy()
        X_t.iloc[:,index] = np.random.permutation(X_t.iloc[:, index])
        return_dict[k] = model.predict(X_t)
    tmp.append(return_dict.values())

    with open(fname, "a") as f:
        f.write(str(indexname))
        f.write(",")
        f.write(','.join(map(str, np.mean(tmp,axis=0)[0])))
        f.write("\n")
        f.flush()


def predict(model, test_df, repeat=10):
    pred = model.predict(test_df)


    return pred



if __name__ =="__main__":
    outputfolder = Parameter.p["folders"]["result_file_folder"]+Parameter.p["challenge"]
    featurefilePath = outputfolder+"/features/"
    featureindex = pd.DataFrame.from_csv(featurefilePath+"all_features.csv", index_col=0)
    print featureindex.iloc[0]['name']
