
import subprocess

subprocess.run(['mkdir', 'metrics'])

models = ['BASEBERT', 'DISTILBERT', 'ROBERTA', 'ALBERT', 'GPT2']
rates = ['0.00001',  '0.0001', '0.001', '0.01', '0.1', '1']

# 5.5 hours estimated on Dr. Henry PC
# 208.3 hours estimated on my PC

for i in models:
    for j in rates:
        subprocess.run(['nohup', 'python3', 'experiment.py', i, j])
