import json
import shutil
import sys

if len(sys.argv) > 1:
    filepath = "/batchfiles/" + sys.argv[1]
    filename = sys.argv[1]
else:
    filename = 'default_parameter_random8_test_extra.json'
    filepath = filename

class Parameter():
    _filename = filename
    with open('../Data/Parameters/'+filepath) as data_file:
        print "load parameters...", filename
        p = json.load(data_file)

    with open(p["folders"]["result_file_folder"]+p["challenge"]+"/"+filename, 'w') as outfile:
        json.dump(p, outfile)