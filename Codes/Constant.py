#-*- coding: utf-8 -*-
import pandas as pd
from JSON_Parameters import Parameter
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def loadLibfmFile_validation(train_libfm, validation_libfm, iter, t_upper=None, t_lower=None):
    trainDF = libfmFileToDF(train_libfm)
    if t_upper != None and t_lower != None:
        trainDF = trainDF[(trainDF['Synergy score'] <= t_lower) | (trainDF['Synergy score'] > t_upper)]
    synScore = trainDF['Synergy score']
    trainDF = trainDF.drop('Synergy score', axis=1)


    validationDF = libfmFileToDF(validation_libfm)
    if t_upper != None and t_lower != None:
        validationDF = validationDF[(validationDF['Synergy score'] <= t_lower) | (validationDF['Synergy score'] > t_upper)]
    validation_synScore = validationDF['Synergy score']
    validationDF = validationDF.drop('Synergy score', axis=1)

    diff = len(trainDF.columns) - len(validationDF.columns)
    if diff > 0:
        for c in range(len(validationDF.columns), len(validationDF.columns) + diff):
            validationDF[c] = [0] * len(validationDF.index)
    if diff < 0:
        for c in range(len(trainDF.columns), len(trainDF.columns) + diff):
            trainDF[c] = [0] * len(trainDF.index)

    # ess_by_fold = dict()
    if Parameter.p["features"]["expected_synergy_scores"]["bool"]:
        essdf = pd.read_csv(Parameter.p["folders"]["feature_file_folder"]+Parameter.p["features"]["expected_synergy_scores"]["folderpath"]+str(iter)+"ess.csv", index_col=0)

        #load ess feature index
        feature_index = list()
        for groupby in Parameter.p["features"]["expected_synergy_scores"]["group_by"]:
            if groupby["bool"] == 1:
                feature_index.extend(list(pd.read_csv(Parameter.p["folders"]["result_file_folder"]+Parameter.p["challenge"]+"/"+Parameter.p["folders"]["feature_index"]+"ExpectedSynergyScore_"+groupby["name"]+".csv", index_col=0).index.values))

        selected_essdf_t = essdf.loc[trainDF.index]
        selected_essdf_v = essdf.loc[validationDF.index]
        selected_essdf_t.columns = trainDF.ix[:,feature_index].columns
        selected_essdf_v.columns = validationDF.ix[:,feature_index].columns
        # print selected_essdf_t.values
        trainDF.ix[:,feature_index] = selected_essdf_t#.values
        validationDF.ix[:,feature_index] = selected_essdf_v#.values


    #normalization
    # trainDF, validationDF = normalization(trainDF, validationDF)

    print len(trainDF.columns), len(validationDF.columns)
    return trainDF, validationDF, synScore, validation_synScore

def libfmFileToDF(path):
    return pd.read_csv(path, index_col=0)


def write(predictedSynergy, answerSetFilePath, rootPath, iter, key = 0, mda=0, fli = 0):
    fr = open(answerSetFilePath,'r')
    currfolder = Parameter.p["folders"]["prediction"]
    if mda == 1:
        currfolder = "/etc/mda/"+str(key)
    if fli == 1:
        currfolder = "etc/feature_layer_importance/"+str(key).replace(".csv","")

    outputfilename = currfolder + str(iter) + Parameter.p["output_files"]["prediction"]
    predFilePath = rootPath + outputfilename

    fw = open(predFilePath,'w')
    lines = fr.readlines()
    i = 0
    fw.write("CELL_LINE,COMBINATION_ID,PREDICTION\n")

    header = lines[0].strip().split(",")
    for h in range(len(header)):
        if header[h] == 'CELL_LINE':
            cellline_index = h
        elif header[h] == 'COMBINATION_ID':
            drugcombination_index = h

    for line in lines:
        splited = line.split(",")
        if i > 0:
            fw.write(splited[cellline_index] + "," + splited[drugcombination_index].replace('\n', '') + "," + str(predictedSynergy[i - 1]) + "\n")
        i+=1

    fw.close()
    fr.close()
    if Parameter.p["experiment_type"] == "train-test":
        import shutil
        shutil.copyfile(predFilePath, "D:/"+Parameter._filename+".csv")
    print "predFilePath", predFilePath
    return predFilePath


def calculate_corr(obsFile, predFile):
    obsdf = pd.read_csv(obsFile,index_col=None)
    preddf = pd.read_csv(predFile,index_col=None)

    return np.corrcoef(obsdf["SYNERGY_SCORE"], preddf["PREDICTION"])[0][1]

def calculate_accuracy(obsFile, predFile, threshold_upper=0, threshold_lower=0):
    from sklearn.metrics import precision_recall_fscore_support

    obsdf = pd.read_csv(obsFile,index_col=None)

    obsdf.loc[obsdf['SYNERGY_SCORE'] <= threshold_lower, 'SYNERGY_SCORE'] = 0
    obsdf.loc[obsdf['SYNERGY_SCORE'] > threshold_upper, 'SYNERGY_SCORE'] = 1

    preddf = pd.read_csv(predFile,index_col=None)

    return zip(["Precision:","Recall:","F1 score:"], list(precision_recall_fscore_support(obsdf['SYNERGY_SCORE'], preddf['PREDICTION'], average='binary'))[:3])


def calculate(obsFile, predFile, threshold_upper=None, threshold_lower=None):
    if Parameter.p["machinelearning_method"].endswith("C"):
        return calculate_accuracy(obsFile, predFile,threshold_upper, threshold_lower)
    else:
        return "correlation coefficient:", calculate_corr(obsFile, predFile), "RMSE:", rmse(obsFile, predFile)

def rmse(obsfp, predfp):
    obs = pd.read_csv(obsfp, index_col=None)
    pred = pd.read_csv(predfp, index_col=None)
    rms = sqrt(mean_squared_error(obs['SYNERGY_SCORE'], pred['PREDICTION']))
    return rms
