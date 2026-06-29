import pandas as pd
import pickle
from pathlib import Path
import yaml
import numpy as np

here = Path(__file__).resolve().parent
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

classificationResultsPath = here / 'checkpoints/classificationResults.pkl'
with open(classificationResultsPath, 'rb') as file:
    classificationResults = pickle.load(file)

dfDict = {}
for dataset in config['DATASETS']:
    dfDict[dataset] = {}
    for mech in config['MISSINGNESS']['mechs']:
        dfDict[dataset][mech] = {}
        for classifier in config['MODELS']['CLASSIFIER']:
            for model in config['MODELS']['EMBEDDING']:
                dfDict[dataset][mech][(classifier, model)] = {}
                for metric in ['accuracy', 'f1_score']:
                    for rate in config['MISSINGNESS']['rates']:
                        metricVals = []
                        for dataDict in classificationResults[dataset][model][mech][rate][classifier]:
                            metricVals.append(dataDict[metric])
                        mean, std = np.mean(metricVals), np.std(metricVals)
                        dfDict[dataset][mech][(classifier, model)][(metric, rate)] = f'${mean:.2f} \pm {std:.2f}$'

with open(here / 'checkpoints/classificationResults.tex', 'w') as file:    
    for dataset in dfDict.keys():
        for mech in config['MISSINGNESS']['mechs']:
            dfDict[dataset][mech] = pd.DataFrame.from_dict(dfDict[dataset][mech], orient='index')
            latexText = dfDict[dataset][mech].to_latex(escape=False)
            file.write(latexText)