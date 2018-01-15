import pandas as pd
from collections import defaultdict
from JSON_Parameters import Parameter
import math

resultFolder = '../Results/'
featureStringdict = dict()

def write_featureIdx(featureDictionary, fileName):
    fileName = fileName.replace(' ',"_")

    targetDir = resultFolder+Parameter.p["challenge"]+"/"+Parameter.p["folders"]["feature_index"]+fileName+".csv"
    df= pd.DataFrame(featureDictionary.items(), columns=['index','name'])
    df.to_csv(targetDir, index=False)

def libfm2dict(str, featuresize):
    splited = str.strip().split(" ")
    result = dict()
    for i in range(featuresize):
        result[i] = 0
    for sp in splited[1:]:
        result[int(sp.split(":")[0])] = sp.split(":")[1]
    result['Synergy score'] = splited[0]
    return result

def libfm(rootPath, trainDir, iter):
    filePath = Parameter.p["folders"]["feature_file_folder"]
    feature_parameters = Parameter.p["features"]


    drug_target_param = feature_parameters["drug_target"]

    ic50_param = feature_parameters["ic50"]
    dss_param = feature_parameters["dss"]
    einf_param = feature_parameters["einf"]

    mutation_param = feature_parameters["mutation"]
    cnv_param = feature_parameters["cnv"]
    gex_param = feature_parameters["gex"]
    synthetic_lethality_param = feature_parameters["synthetic_lethality"]


    expected_synergy_score_param = feature_parameters["expected_synergy_scores"]
    additionscore_param = feature_parameters["additionscore"]


    libfm_id=0
    featdict = {}

    if drug_target_param["bool"]:
        drugname_target_dict = defaultdict(list)
        targetid_dict = defaultdict(list)
        df = pd.read_csv(filePath+drug_target_param["filepath"])
        drug_target_set=set()
        for row in df.itertuples():
            if pd.notnull(row[2]):
                targetSplit=row[2].split(',')
                drugname_target_dict[row[1]] = [x.strip() for x in targetSplit]
                for i in range(len(targetSplit)):
                    drug_target_set.add(targetSplit[i].strip())

        excludeIdxList = dict()
        for i,drug_target in enumerate(drug_target_set,start=libfm_id):
            targetid_dict[drug_target] = i
            featdict[i] = "TARGET@"+drug_target
            excludeIdxList[i] = drug_target
        write_featureIdx(excludeIdxList, "target")
        libfm_id+=len(drug_target_set)



    if ic50_param["bool"]:
        ic50Id_diff = libfm_id
        ic50Id_sum = libfm_id+1

        featdict[ic50Id_diff] = "IC50@diff"
        featdict[ic50Id_sum] = "IC50@sum"


        excludeIdxList = dict()
        excludeIdxList[ic50Id_diff] = "ic50_diff"
        excludeIdxList[ic50Id_sum] = "ic50_sum"
        write_featureIdx(excludeIdxList, "ic50")

        libfm_id+=2

    if einf_param["bool"]:
        einfId_diff = libfm_id
        einfId_sum = libfm_id+1

        featdict[einfId_diff] = "einf@diff"
        featdict[einfId_sum] = "einf@sum"


        excludeIdxList = dict()
        excludeIdxList[einfId_diff] = "einf_diff"
        excludeIdxList[einfId_sum] = "einf_sum"
        write_featureIdx(excludeIdxList, "einf")

        libfm_id+=2


    if mutation_param["bool"]:
        mutation_gene_set=set()
        mutation_gene_set_name2list_dict = dict()
        mutation_gene_set_name2object_dict = dict()
        for f in mutation_param["pathways"]:
            if f["bool"] or mutation_param["all"]:
                fr = pd.read_csv(filePath+f["filepath"], index_col=None)
                mutation_gene_set.update(fr["GENES"].values)
                mutation_gene_set_name2list_dict[f["name"]] = set(fr["GENES"].values)
                mutation_gene_set_name2object_dict[f["name"]] = f

        id_dict={}
        for i,gene in enumerate(mutation_gene_set,start=libfm_id):
            id_dict [gene]=i
            featdict[i] = "MUTATION@"+str(gene)

        cell_line_mutation={}
        df2 = pd.read_csv(filePath+mutation_param["filepath"])
        cellmutnamedict = {}
        for row in df2.itertuples():
            gene_name=row[4]
            cl_name = row[1]
            if gene_name in id_dict:
                if gene_name=='BRAF' and row[7]=='p.V600E' and 'BRAF_V600E' in id_dict:
                    if row[5] in cell_line_mutation:
                        cell_line_mutation[cl_name].add(id_dict['BRAF_V600E'])
                        cellmutnamedict[cl_name].add("BRAF_V600E")

                    else:
                        cell_line_mutation[cl_name]=set([id_dict['BRAF_V600E']])
                        tmp = set()
                        tmp.add("BRAF_V600E")
                        cellmutnamedict[cl_name]=tmp


                else:
                    if cl_name in cell_line_mutation:
                        cell_line_mutation[cl_name].add(id_dict[gene_name])
                        cellmutnamedict[cl_name].add(gene_name)

                    else:
                        cell_line_mutation[cl_name]=set([id_dict[gene_name]])
                        tmp = set()
                        tmp.add(gene_name)
                        cellmutnamedict[cl_name] = tmp


        drug_target_not_in_cell_line=defaultdict(set)
        for cell_line_var in cell_line_mutation:
            for drug_target_id_var in id_dict.values():
                if drug_target_id_var not in cell_line_mutation[cell_line_var]:
                    drug_target_not_in_cell_line[cell_line_var].add(drug_target_id_var)

        excludeIdxList = dict()
        for geneset_name in mutation_gene_set_name2list_dict:
            fr = pd.read_csv(filePath+mutation_gene_set_name2object_dict[geneset_name]["filepath"], index_col=None)
            genes = fr["GENES"].values
            for gene in genes:
                excludeIdxList[id_dict[gene]] = gene
        write_featureIdx(excludeIdxList, "mutation")

        libfm_id+=len(id_dict)



    if cnv_param["bool"]:
        cnvGeneSet = set()
        clcnv_HA=defaultdict(list)
        origin_files = []

        for f in cnv_param["pathways"]:
            if f["bool"] or cnv_param["all"]:
                origin_files.append(f["filepath"])
        excludeIdxList = dict()
        for origin_file in list(set(origin_files)):
            pr=pd.read_csv(filePath + origin_file, index_col=0)

            cnvTmp = pr.columns
            for v in cnvTmp:
                cnvGeneSet.add(v)

            cnvId={}

            for i,name in enumerate(cnvTmp,start=libfm_id):
                cnvId[name]=i
                featdict[i] = "CNV_" + origin_file.replace("cnv/","").replace("1-1_twocol.csv","") + "@" + name
                excludeIdxList[i] = name

            libfm_id+=len(cnvId)

            for col in cnvTmp:
                for cellLine in pr.index:
                    if pr.loc[cellLine,col]:clcnv_HA[cellLine].append(str(cnvId[col])+":"+str(pr.loc[cellLine,col]))
        write_featureIdx(excludeIdxList,"cnv")

    if additionscore_param["bool"]:
        pr=pd.read_csv(filePath+additionscore_param["filepath"],index_col=None)
        cladditionscore={}
        for row in pr.itertuples():
            tmp=''
            tmp+=' '+str(libfm_id)+':'+str(row[3])
            cladditionscore[row[1]+"."+row[2]]=tmp

        featdict[libfm_id] = "ADDITION_SCORE"
        excludeIdxList = dict()
        excludeIdxList[libfm_id] = "ADDITION_SCORE"
        write_featureIdx(excludeIdxList, "additionscore")
        libfm_id+=1

    if gex_param["bool"]:
        gex = pd.read_csv(filePath+gex_param["filepath"], index_col=0)
        genesets = []
        for f in gex_param["pathways"]:
            if f["bool"] or gex_param["all"]:
                genesets.append(f["filepath"])

        cellline_gex_dict = {}
        excludeIdxList = dict()
        for geneset in genesets:
            geneIDs = set()
            readFile = open(filePath + geneset, "r")
            with open(filePath + geneset) as f:
                lines = f.read().splitlines()
            geneIDs.update(set(lines[1:]))
            #only use the genes in the gex file
            real_geneIDs = [geneID for geneID in geneIDs if geneID in gex.columns]

            tmp_cellline_gex_dict = gex.loc[:,real_geneIDs].transpose().to_dict(orient='list')

            for key, value in tmp_cellline_gex_dict.iteritems():
                tempStr = ''
                for i, v in enumerate(value, start = libfm_id):
                    tempStr += ' '+str(i) + ":" + str(v)
                if key in cellline_gex_dict:
                    cellline_gex_dict[key] += tempStr
                else:
                    cellline_gex_dict[key] = tempStr

            for i in range(len(real_geneIDs)):
                featdict[libfm_id+i] = "GEX_"+geneset.replace(".txt","").replace("gex/genesets/","")+"@"+str(real_geneIDs[i])
                excludeIdxList[libfm_id+i] = str(real_geneIDs[i])
            libfm_id += len(real_geneIDs)
        write_featureIdx(excludeIdxList,"gex")


    if dss_param["bool"]:
        dss_df = pd.read_csv(filePath+dss_param["filepath"])
        dss_dict={}
        for i in range(len(dss_df.index)):
           dss_dict[dss_df.iloc[i,0]+":"+dss_df.iloc[i,1]]= dss_df.iloc[i,2]

        dssId_diff = libfm_id
        dssId_sum = libfm_id+1
        featdict[dssId_diff] = "dss@diff"
        featdict[dssId_sum] = "dss@sum"
        excludeIdxList = dict()
        excludeIdxList[dssId_diff] = "dss_diff"
        excludeIdxList[dssId_sum] = "dss_sum"

        write_featureIdx(excludeIdxList, "dss")
        libfm_id+=2


    if expected_synergy_score_param["bool"]:
        cal_methods = list()
        for f in expected_synergy_score_param["methods"]:
            if f["bool"] or expected_synergy_score_param["all"]:
                cal_methods.append(f["name"])

        for groupby_method in expected_synergy_score_param["group_by"]:
            excludeIdxList = dict()
            if groupby_method["bool"]:
                for cal_method in cal_methods:
                    featdict[libfm_id] = "ExpectedSynergyScore@"+cal_method+"_"+groupby_method["name"]
                    excludeIdxList[libfm_id] = cal_method+"_"+groupby_method["name"]
                    libfm_id += 1
                write_featureIdx(excludeIdxList, "ExpectedSynergyScore_"+groupby_method["name"])



    if synthetic_lethality_param["bool"]:
        sldf = pd.read_csv(filePath+synthetic_lethality_param["filepath"])

        sldict_score = dict()
        sldict_count = dict()
        for i,name in enumerate(set(sldf.index)):
            if sldf.iloc[i]['CELL_LINE.DRUG'] in sldict_score:
                sldict_score[sldf.iloc[i]['CELL_LINE.DRUG']] += sldf.iloc[i]['SCORE']
                sldict_count[sldf.iloc[i]['CELL_LINE.DRUG']] += 1
            else:
                sldict_score[sldf.iloc[i]['CELL_LINE.DRUG']] = sldf.iloc[i]['SCORE']
                sldict_count[sldf.iloc[i]['CELL_LINE.DRUG']] = 1

        excludeIdxList = dict()
        slid_score = libfm_id
        slid_count = libfm_id + 1
        featdict[slid_score] = "synthetic_lethality_score"
        excludeIdxList[slid_score] = "synthetic_lethality_score"
        featdict[slid_count] = "synthetic_lethality_count"
        excludeIdxList[slid_count] = "synthetic_lethality_count"

        write_featureIdx(excludeIdxList, "synthetic_lethality")
        libfm_id += 2



    ############################################################################################################################

    write_featureIdx(featdict, "all_features")


    print 'Feature Size',libfm_id
    libfm_fold=rootPath+Parameter.p["folders"]["libfm"]+str(iter)+Parameter.p["output_files"]["libfm_fold"]


    def makeLibfmInput(file_directory, libfm_file, score):
        df=pd.read_csv(file_directory, index_col=0)

        with open(libfm_file,'w') as fw:
            row_id = 0
            cellline_index = 0
            drug_a_index = 0
            drug_b_index = 0
            drugcombination_index = 0
            synergyscore_index = 0
            ic50_a_index= 0
            ic50_b_index= 0
            maxconc_a_index = 0
            maxconc_b_index = 0
            einf_a_index = 0
            einf_b_index = 0

            for j in range(len(df.columns)):
                if df.columns[j] == Parameter.p["synergyscore_methods"]:
                    synergyscore_index = j + 1
                    Parameter.p["synergyscore_index"] = j
                elif df.columns[j] == 'COMPOUND_A':
                    drug_a_index = j + 1
                    Parameter.p["drug_a_index"] = j
                elif df.columns[j] == 'COMPOUND_B':
                    drug_b_index = j + 1
                    Parameter.p["drug_b_index"] = j
                elif df.columns[j] == 'CELL_LINE':
                    cellline_index = j + 1
                    Parameter.p["cellline_index"] = j
                elif df.columns[j] == 'COMBINATION_ID':
                    drugcombination_index = j + 1
                    Parameter.p["drugcombination_index"] = j
                elif df.columns[j] == 'IC50_A':
                    ic50_a_index = j + 1
                    Parameter.p["ic50_a_index"] = j
                elif df.columns[j] == 'IC50_B':
                    ic50_b_index = j + 1
                    Parameter.p["ic50_b_index"] = j
                elif df.columns[j] == 'MAX_CONC_A':
                    maxconc_a_index = j + 1
                    Parameter.p["maxconc_a_index"] = j
                elif df.columns[j] == 'MAX_CONC_B':
                    maxconc_b_index = j + 1
                    Parameter.p["einf_a_index"] = j
                elif df.columns[j] == 'Einf_A':
                    einf_a_index = j + 1
                    Parameter.p["einf_a_index"] = j
                elif df.columns[j] == 'Einf_B':
                    einf_b_index = j + 1
                    Parameter.p["einf_b_index"] = j

            for row in df.itertuples():
                featureString=''
                row_index = int(df.index[row_id])
                synergy_score=row[synergyscore_index]
                drug_a=row[drug_a_index]
                drug_b=row[drug_b_index]
                cell_line_name=row[cellline_index]
                drug_combination = row[drugcombination_index]

                if score: featureString+=str(synergy_score)
                else: featureString+='0'

                if drug_target_param["bool"]:
                    u=set(drugname_target_dict[drug_a]).union(set(drugname_target_dict[drug_b]))
                    for x in u:
                        val = 1
                        featureString+=' '+str(targetid_dict[x])+':'+str(val)

                if ic50_param["bool"]:
                    featureString+=' '+str(ic50Id_diff)+':'+str(math.log(abs(row[ic50_a_index]- row[ic50_b_index])+1,2))+' '+ str(ic50Id_sum)+':'+str(math.log((row[ic50_a_index] + row[ic50_b_index])+1,2))

                if einf_param["bool"]:
                    einf_1 = float(row[einf_a_index])
                    einf_2 = float(row[einf_b_index])

                    # featureString+=" "+str(einfId_a)+':'+str(einf_1)
                    # featureString+=" "+str(einfId_b)+':'+str(einf_2)
                    featureString+=" "+str(einfId_diff)+':'+str(abs(einf_1 - einf_2))
                    featureString+=" "+str(einfId_sum)+':'+str(einf_1 + einf_2)


                if mutation_param["bool"]:
                    if cell_line_name in cell_line_mutation:
                        for x in cell_line_mutation[cell_line_name]: featureString+=' '+ str(x)+':1'
                    # print "mut"

                if cnv_param["bool"]:
                    if cell_line_name in clcnv_HA: featureString+=' '+' '.join(clcnv_HA[cell_line_name])
                if additionscore_param["bool"]:
                    if cell_line_name+"."+drug_combination in cladditionscore: featureString+=cladditionscore[cell_line_name+"."+drug_combination]
                if gex_param["bool"]:
                    if cell_line_name in cellline_gex_dict:
                        featureString+=cellline_gex_dict[cell_line_name]
                if dss_param["bool"]:

                    key1 = cell_line_name+":"+drug_a
                    key2 = cell_line_name+":"+drug_b
                    val1 = dss_dict[key1]
                    val2 = dss_dict[key2]
                    featureString+= ' '+str(dssId_diff)+":"+str(abs(val1-val2))
                    featureString+= ' '+str(dssId_sum)+":"+str(val1+val2)

                if synthetic_lethality_param["bool"]:
                    val1 = 0.0
                    val2 = 0.0
                    count1 = 0.0
                    count2 = 0.0
                    if cell_line_name + "."+drug_a in sldict_score:
                        val1 = sldict_score[cell_line_name + "."+drug_a]
                        count1 = sldict_count[cell_line_name + "."+drug_a]
                    elif cell_line_name + "."+drug_b in sldict_score:
                        val2 = sldict_score[cell_line_name + "."+drug_b]
                        count2 = sldict_count[cell_line_name + "."+drug_b]
                    featureString += ' '+ str(slid_score)+":"+str(val1+val2)
                    featureString += ' '+ str(slid_count)+":"+str(count1+count2)

                ########################################################################################
                featureString=featureString.strip()+'\n'
                tmpfeaturedict = libfm2dict(featureString, libfm_id)
                featureStringdict = dict()
                featureStringdict[row_index] = tmpfeaturedict
                rdf = pd.DataFrame.from_dict(featureStringdict)
                if row_id == 0:
                    rdf.transpose().to_csv(libfm_file, mode='w', header=True)
                else:
                    rdf.transpose().to_csv(libfm_file, mode='a', header=False)
                row_id+=1

    makeLibfmInput(trainDir, libfm_fold, True)


    return libfm_fold