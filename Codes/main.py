from Constant import *
from makeSingleFeature import *
from Functions import *
from multiprocessing import Process, Manager
import makeExpectedSS as ess
from JSON_Parameters import Parameter
import shutil
import time
data_folder_path = "../Data/"
result_path = "../Results/"

load = False

try:
    t_upper = Parameter.p["classification_threshold"][0]
    t_lower = Parameter.p["classification_threshold"][1]
except:
    t_upper = None
    t_lower = None


def converter(val, threshold_upper, threshold_lower):
    if val>threshold_upper:
        return 1
    elif val<threshold_lower:
        return 0

def converter_df(df, threshold_upper, threshold_lower):

    df.loc[df['SYNERGY_SCORE'] <= threshold_lower, 'SYNERGY_SCORE'] = 0
    df.loc[df['SYNERGY_SCORE'] > threshold_upper, 'SYNERGY_SCORE'] = 1
    return df

def make_model_and_predict(train_df, synergy_score, test_df, synergy_score_validation, C, gamma, featurenames = None):
    if Parameter.p["machinelearning_method"] == "SVR":
        model = make_svr_model(train_df, synergy_score, C = C, gamma = gamma)
    elif Parameter.p["machinelearning_method"] == "SVRpoly":
        model = make_svr_model(train_df, synergy_score, C = C, gamma = gamma, poly=True)
    elif Parameter.p["machinelearning_method"] == "RF":
        model = make_rf_model(train_df, synergy_score, n_tree = C, max_features=gamma)
    elif Parameter.p["machinelearning_method"] == "Extra":
        model = make_extra_model(train_df, synergy_score, n_tree = C, max_features=0.33)
    elif Parameter.p["machinelearning_method"] == "Ada":
        model = make_ada_model(train_df, synergy_score, n_tree = C, base_estimator=gamma)
    elif Parameter.p["machinelearning_method"] == "Ridge":
        model = make_ridge_model(train_df, synergy_score, alpha = C)
    elif Parameter.p["machinelearning_method"] == "Lasso":
        model = make_lasso_model(train_df, synergy_score, alpha = C)
    elif Parameter.p["machinelearning_method"] == "Elasticnet":
        model = make_elasticnet_model(train_df, synergy_score, alpha = C, l1_ratio = gamma)
    elif Parameter.p["machinelearning_method"] == "SGD":
        model = make_sgd_model(train_df, synergy_score, alpha = C)
    elif Parameter.p["machinelearning_method"] == "KernelRidge":
        model = make_kernelridge_model(train_df, synergy_score, alpha = C)
    elif Parameter.p["machinelearning_method"] == "NN":
        model = make_nn_model(train_df, synergy_score, nlayer = C)
    elif Parameter.p["machinelearning_method"] == "SVC":
        model = make_svc_model(train_df, synergy_score, kernel=gamma, C=C)
    elif Parameter.p["machinelearning_method"] == "SVCpolyC":
        model = make_svc_model(train_df, synergy_score, kernel='poly', C=C, degree=gamma)
    elif Parameter.p["machinelearning_method"] == "DTC":
        model = make_dtc_model(train_df, synergy_score)
    elif Parameter.p["machinelearning_method"] == "RFC":
        model = make_rfc_model(train_df, synergy_score, n_estimators = C, max_features = gamma)
    elif Parameter.p["machinelearning_method"] == "ETC":
        model = make_etc_model(train_df, synergy_score, n_estimators = C, max_features = gamma)
    elif Parameter.p["machinelearning_method"] == "LRC": #logistic regression
        model = make_lrc_model(train_df, synergy_score, C, gamma)

    pred = predict(model, test_df)
    return pred

def do(train_file_path, test_file_path, cvtrain_file_path, cvtest_file_path, challenge, iter, C, gamma):

    curr_result_path = result_path+challenge+"/"
    result_file_name = "result_C_"+str(C)+".txt"

    """
    1. make libfm file(exclude predefined indexes)
    2. load libfm file to dataframe
    3. SVR fit
    4. make result file
    5. 10 cv for train set
    6. make confidence file
        """

    # #1 make libfm file format
    curr_libfmfolder = Parameter.p["folders"]["result_file_folder"]+Parameter.p["challenge"]+"/"+Parameter.p["folders"]["libfm"]

    if Parameter.p["experiment_type"] == "train-validation":
        train_libfm_path = curr_libfmfolder+str(iter)+Parameter.p["output_files"]["libfm_train"]
        test_libfm_path = curr_libfmfolder+str(iter)+Parameter.p["output_files"]["libfm_validation"]
    elif Parameter.p["experiment_type"] == "train-test":
        iter = ''
        train_libfm_path = curr_libfmfolder+str(iter)+Parameter.p["output_files"]["libfm_finaltrain"]
        test_libfm_path = curr_libfmfolder+str(iter)+Parameter.p["output_files"]["libfm_finaltest"]


    #2 load libfm file as dataframe

    if Parameter.p["machinelearning_method"].endswith("C"):
        train_df, test_df, synergy_score, synergy_score_validation = loadLibfmFile_validation(train_libfm_path, test_libfm_path, iter, t_upper, t_lower)
        synergy_score = [converter(x, t_upper, t_lower) for x in synergy_score]
        synergy_score_validation = [converter(x, t_upper, t_lower) for x in synergy_score_validation]
    else:
        train_df, test_df, synergy_score, synergy_score_validation = loadLibfmFile_validation(train_libfm_path, test_libfm_path, iter)
    outputfolder = Parameter.p["folders"]["result_file_folder"]+Parameter.p["challenge"]
    featurenames = pd.read_csv(outputfolder+"/features/all_features.csv",index_col=0)


    #3 predict
    if Parameter.p["prediction"] == 1:
        print "predicting...", Parameter.p["machinelearning_method"]
        print len(test_df.index)
        #4 predict synergy scores
        pred = make_model_and_predict(train_df, synergy_score, test_df, synergy_score_validation, C, gamma, list(featurenames['name'].values))

        #3-1 save the result
        if challenge == "1A" or challenge == "1B":
            pred_file_path = write(pred, test_file_path, curr_result_path, iter)

            if Parameter.p["experiment_type"] == "train-validation":
                fname = "/validation"+str(iter)+".csv"
            elif Parameter.p["experiment_type"] == "train-test":
                fname = "/finaltest.csv"

            tmp_test_file_path = outputfolder+"/"+Parameter.p["folders"]["answer"]+fname
            score = calculate(tmp_test_file_path, pred_file_path, t_upper, t_lower)
            print str(iter)+","+ ' '.join(map(str,list(score)))



def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def rmdir(directory):
    try:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        else:
        # if os.path.exists(directory):
            os.remove(directory)
    except:
        pass

def make_output_directory(directory, challenge):
    curr_result_path = result_path+challenge+"/"
    mkdir(result_path)
    mkdir(curr_result_path)
    for f in os.listdir(curr_result_path):
        rmdir(curr_result_path+f)
    time.sleep(8)


    mkdir(curr_result_path+"/features")

    mkdir(curr_result_path+"/etc")
    mkdir(curr_result_path+"/etc/10cv")
    mkdir(curr_result_path+"/etc/libfm")
    mkdir(curr_result_path+"/etc/prediction")
    mkdir(curr_result_path + "/etc/answer")
    return curr_result_path

def save(p):

    #copy answer and chanege column name as 'synergy score'
    for f in range(p["iteration"]):
        if p["experiment_type"] == "train-validation":
            fname = "/validation"+str(f)+".csv"
        elif p["experiment_type"] == "train-test":
            fname = "/finaltest.csv"


        answer = pd.read_csv(p["folders"]["data_file_folder"]+p["datatype"]+fname, index_col=0)
        thecol = 0
        for col in range(len(answer.columns)):
            if answer.columns[col] == p["synergyscore_methods"]:
                thecol = col
        newcolumns = answer.columns.values
        newcolumns[thecol] = 'SYNERGY_SCORE'
        answer.columns = newcolumns
        answer = converter_df(answer, t_upper, t_lower)
        answer.to_csv(p["folders"]["result_file_folder"]+p["challenge"]+"/"+p["folders"]["answer"]+fname, index = None)

def merge_and_divide_libfm():
    curr_folder = Parameter.p["folders"]["result_file_folder"]+Parameter.p["challenge"]+"/"+Parameter.p["folders"]["libfm"]
    mergedfile = Parameter.p["output_files"]["libfm_merged"]

    #merge
    values_by_line = dict()
    with open(curr_folder+mergedfile, "w") as f:
        header = ''
        for i in os.listdir(curr_folder):
            if i != mergedfile:
                c = open(curr_folder+i, 'r')

                lines = c.readlines()
                header = lines[0]
                for line in lines[1:]:
                    splited = line.split(",")
                    values_by_line[int(splited[0])] = line
                c.close()

        f.write(header)
        for key, value in values_by_line.iteritems():
            f.write(str(key))
            f.write(",")
            f.write(value)


    #select and write
    def select_and_write(key):
        sample_folder = Parameter.p["folders"]["data_file_folder"] + datatype +"/"
        for i in range(Parameter.p["iteration"]):
            if Parameter.p["experiment_type"] =="train-validation":
                tmpdf = pd.read_csv(sample_folder+Parameter.p["input_files"][key+"_file_path"]+str(i)+".csv", index_col=0)
            else:
                i = ''
                tmpdf = pd.read_csv(sample_folder+Parameter.p["input_files"][key+"_file_path"]+".csv", index_col=0)
            selected_lines = [values_by_line[k] for k in tmpdf.index]
            with open(curr_folder+str(i)+Parameter.p["output_files"]["libfm_"+key], "w") as c:
                c.write(header)
                for s in range(len(selected_lines)):
                    c.write(selected_lines[s])


    if Parameter.p["experiment_type"] =="train-validation":
        select_and_write("train")
        select_and_write("validation")
    elif Parameter.p["experiment_type"] =="train-test":
        select_and_write("finaltrain")
        select_and_write("finaltest")

if __name__ == '__main__':
    challenge = Parameter.p["challenge"]
    datatype = Parameter.p["datatype"]

    experiment_type = Parameter.p["experiment_type"]

    if experiment_type == "train-validation":
        iteration = Parameter.p["iteration"]
    elif experiment_type == "train-test":
        iteration = 1

    Clist = Parameter.p["Clist"]
    gammalist = Parameter.p["gammalist"]
    if load == False:
        print "make output directories..."
        curr_result_path = make_output_directory(result_path, challenge)
        result_file_name = ""

        print "save default settings..."
        save(Parameter.p)

        folditeration = 10

        #make expected synergy scores feature
        if Parameter.p["features"]["expected_synergy_scores"]["bool"]:
            print "make expected synergy scores.."
            if Parameter.p["experiment_type"] =="train-validation":
                for iter in range(iteration):
                    train_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["train_file_path"]+str(iter)+".csv"])
                    ess.make_simple_expected_synergy_score(train_file_path, Parameter.p["folders"]["feature_file_folder"]+Parameter.p["features"]["expected_synergy_scores"]["folderpath"]+str(iter), iter)
            elif Parameter.p["experiment_type"] =="train-test":
                iter = ''
                train_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["finaltrain_file_path"]+".csv"])
                ess.make_simple_expected_synergy_score(train_file_path, Parameter.p["folders"]["feature_file_folder"]+Parameter.p["features"]["expected_synergy_scores"]["folderpath"]+str(iter), iter)
            elif Parameter.p["experiment_type"] =="practrain-practest":
                iter = ''
                train_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["practrain_file_path"]+".csv"])
                ess.make_simple_expected_synergy_score(train_file_path, Parameter.p["folders"]["feature_file_folder"]+Parameter.p["features"]["expected_synergy_scores"]["folderpath"]+str(iter), iter)

        # divide all samples into 10 folds
        totaldf = pd.read_csv(Parameter.p["input_files"]["total_file_path"], index_col=0)
        unitsize = len(totaldf.index) // folditeration
        for iter in range(folditeration):
            file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], Parameter.p["input_files"]["fold_files_path"], Parameter.p["input_files"]["validation_file_path"]+str(iter)+".csv"])
            if iter < 9:
                totaldf.iloc[iter*unitsize: (iter+1)*unitsize].to_csv(file_path)
            else:
                totaldf.iloc[iter*unitsize:].to_csv(file_path)

        # make libfm file for every sample
        processes = []
        for iter in range(folditeration):
            file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], Parameter.p["input_files"]["fold_files_path"], Parameter.p["input_files"]["validation_file_path"]+str(iter)+".csv"])
            p = Process(target=libfm, args=(curr_result_path, file_path, iter))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        merge_and_divide_libfm()


    if experiment_type == "train-validation":
        for C in Clist:
            processes = []
            result_file_name = "result_C_"+str(C)+".txt"
            f = open(curr_result_path+Parameter.p["folders"]["score"]+result_file_name,"w")
            f.close()
            for gamma in gammalist:
                for iter in range(iteration):
                    print "Paremters", C, gamma

                    train_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["train_file_path"]+str(iter)+".csv"])
                    validation_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["validation_file_path"]+str(iter)+".csv"])

                    print "Train: ", train_file_path
                    print "Validation: ", validation_file_path

                    cvtrain_file_path = train_file_path
                    cvtest_file_path = validation_file_path

                    p = Process(target=do, args=(train_file_path, validation_file_path, cvtrain_file_path, cvtest_file_path, challenge, iter, C, gamma))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

    elif experiment_type == "train-test":
        train_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["finaltrain_file_path"]+".csv"])
        test_file_path = "/".join([data_folder_path + Parameter.p["input_files"]["input_files_path"], datatype, Parameter.p["input_files"]["finaltest_file_path"]+".csv"])

        print "Train: ", train_file_path
        print "Validation: ", test_file_path

        cvtrain_file_path = train_file_path
        cvtest_file_path = test_file_path
        for C in Clist:
            for gamma in gammalist:
                do(train_file_path, test_file_path, cvtrain_file_path, cvtest_file_path, challenge, '', C, gamma)

