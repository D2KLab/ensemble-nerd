import subprocess
import sys

base = sys.argv[1]
cmd_features = ['python3','getDatasetFeatures.py', base]


try:
    lang = sys.argv[2]
    cmd_features.append('--lang')
    cmd_features.append(lang)
except:
    lang = 'GUESS'


cmd_train_ENTTR = ['python3','ENTTR_train.py', base]
cmd_train_ENND = ['python3','ENND_train.py', base]
cmd_output = ['python3','getOutput.py',base,base]
cmd_extractors = ['python3','getExtractorTypesNormalization.py',base]
cmd_evaluate = ['python3','evaluateDatasetResults.py',base]

if lang == 'JUMP':
	print('jumping features')
	commands = [cmd_train_ENTTR,cmd_train_ENND,cmd_output,cmd_extractors,cmd_evaluate]
else:
	commands = [cmd_features,cmd_train_ENTTR,cmd_train_ENND,cmd_output,cmd_extractors,cmd_evaluate]

for cmd in commands:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print (line)
    p.wait()
    print (p.returncode)