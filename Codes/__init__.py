

feature_list = {
'target' : range(0,99),
'ic50':range(99,103),
'mutation':range(103,416),
'einf':range(416, 420),
'prerher2':range(420,423),
'tspair':range(423, 437),
'cnv':range(437, 493),
'additionscore':[493],
'cbioportalgex':range(494, 679),
'maxcon':range(679, 683),
'dss':range(683, 687),
'emt':[687],
'rule':range(688, 692),
'gsea':range(692, 742),
'fingerprint':range(742, 904),
'go':range(904, 3101),
'mmf_mean':range(3101, 3109),
'mmf_median':range(3109,3117),
'mmf_size':range(3117,3125),
'mmf_std':range(3125,3133),
'four':range(3133, 3281)
}

for key in feature_list.keys():
    with open("../Results/Bonus/excluded/%s.txt"%key , 'w') as fw:
        for v in feature_list[key]:
            fw.write(str(v) + '\n')
