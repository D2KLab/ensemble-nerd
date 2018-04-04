import subprocess
import sys

base = sys.argv[1]
cmd_features = ['getDatasetFeatures.py', base]

try:
    lang = sys.argv[2]
    cmd_features.append('--lang')
    cmd_features.append(lang)
except:
    pass

cmd_train_ENTTR = ['ENTTR_train.py', base]
cmd_train_ENND = ['ENND_train.py', base]
cmd_output = ['getOutput.py',base,base]
cmd_evaluate = ['evaluateDatasetResults.py',base]

commands = [cmd_features,cmd_train_ENTTR,cmd_train_ENND,cmd_output,cmd_evaluate]

for cmd in commands:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        print (line)
    p.wait()
    print (p.returncode)