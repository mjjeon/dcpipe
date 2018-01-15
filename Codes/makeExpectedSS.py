import pandas as pd
import numpy as np
from JSON_Parameters import Parameter
from operator import add

def make_simple_expected_synergy_score(inputfilename, outputfilename, iter):
	colnames = ['CELL_LINE','COMPOUND_B','COMPOUND_A','MAX_CONC_B','MAX_CONC_A','IC50_B','H_B','Einf_B','IC50_A','H_A','Einf_A','SUM_SYN_ANT_LOEWE','SUM_SYN_ANT_BLISS','SUM_SYN_ANT_HSA','COMBINATION_ID']
	df = pd.read_csv(inputfilename, index_col = 0).loc[:,colnames]

	df2 = pd.read_csv(inputfilename, index_col = 0).loc[:,colnames]
	df2.columns = ['CELL_LINE','COMPOUND_B','COMPOUND_A','MAX_CONC_B','MAX_CONC_A','IC50_B','H_B','Einf_B','IC50_A','H_A','Einf_A','SUM_SYN_ANT_LOEWE','SUM_SYN_ANT_BLISS','SUM_SYN_ANT_HSA','COMBINATION_ID']

	merged = pd.concat([df, df2])

	criteria = list()
	for method in Parameter.p["features"]["expected_synergy_scores"]["methods"]:
		for groupby in Parameter.p["features"]["expected_synergy_scores"]["group_by"]:
			if groupby["bool"] == 1:
				if groupby["name"] == "CELL_LINE_COMPOUND_A":
					criteria.append(['CELL_LINE', 'COMPOUND_A'])
				else:
					criteria.append(groupby["name"])

	for col in criteria:
		if len(list(col)) == 2:
			fname = '_'.join(col)
		else:
			fname = col
		for method in ['mean','median','std']:
			if 'COMPOUND_A' in col:
				if method == 'mean':
					tmp = merged.groupby(col).mean()
				elif method == 'median':
					tmp = merged.groupby(col).median()
				elif method == 'std':
					tmp = merged.groupby(col).std()
			else:
				if method == 'mean':
					tmp = df.groupby(col).mean()
				elif method == 'median':
					tmp = df.groupby(col).median()
				elif method == 'std':
					tmp = df.groupby(col).std()

			tmp_norm = (tmp) #/ (tmp.max())
			tmp_norm = tmp_norm.fillna(0)
			tmp_norm.to_csv(outputfilename+fname+"_"+method+".csv")
	if Parameter.p["experiment_type"] == "practrain-practest":
		total = pd.read_csv(Parameter.p["features"]["expected_synergy_scores"]["total_with_prac"], index_col=0)
	else:
		total = pd.read_csv(Parameter.p["features"]["expected_synergy_scores"]["total"], index_col=0)
	newtotal = pd.DataFrame()
	for col in criteria:
		if len(list(col)) == 2:
			fname = '_'.join(col)
			index_col = [0,1]
		else:
			fname = col
			index_col= 0
		for m in Parameter.p["features"]["expected_synergy_scores"]["methods"]:
			if m["bool"]:
				method = m["name"]

				df = pd.read_csv(outputfilename+fname+"_"+method+".csv",index_col=index_col).loc[:,Parameter.p["synergyscore_methods"]]

				if len(list(col)) == 2:
					colB = [x.replace("_A","_B") for x in col]
					tmpdf = df.loc[[tuple(x) for x in total.loc[:,col].values]]
					tmpdf2 = df.loc[[tuple(x) for x in total.loc[:,colB].values]]
					filledtmpdf = tmpdf.fillna(tmpdf.mean())
					filledtmpdf2 = tmpdf2.fillna(tmpdf2.mean())
					a = list(filledtmpdf.values)
					b = list(filledtmpdf2.values)
					c = map(add, a, b)
				else:
					tmpdf = df.loc[total.loc[:,col]]
					filledtmpdf = tmpdf.fillna(tmpdf.mean())
					if col == 'COMPOUND_A':
						colB = col.replace("_A","_B")
						tmpdf2 = df.loc[total.loc[:,colB]]
						filledtmpdf2 = tmpdf2.fillna(tmpdf2.mean())
						a = list(filledtmpdf.values)
						b = list(filledtmpdf2.values)
						c = map(add, a, b)
					else:
						c = list(filledtmpdf.values)

				newtotal.ix[:,'_'.join(map(str, [fname, method]))] = c
	newtotal.index = total.index
	newtotal.fillna(0).to_csv(Parameter.p["folders"]["feature_file_folder"]+Parameter.p["features"]["expected_synergy_scores"]["folderpath"]+str(iter)+"ess.csv")

