#get the options
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from pprint import pprint

def parseFile():
    pathFile = sys.argv[1]
    print("The path file is: " + pathFile)


    scenarioToDiz = {}

    with open(pathFile) as f:
        #read two lines at a time
        for line1, line2 in zip(f, f):
            info = line1.split()
            scenarioName = info[1]
            threshold = info[3]
            if scenarioName not in scenarioToDiz:
                scenarioToDiz[scenarioName] = {}
            scenarioToDiz[scenarioName][threshold] = json.loads(line2)
            #the second line should be read as a json
    #pprint(scenarioToDiz)

    return scenarioToDiz



if __name__ == "__main__":
    info = parseFile()  
    #the sum of n_genuine_faces and n_impostor_faces gives the number of frame. This will be used to do a weighted average
    #for each threshold, I will compute the weighted average of the FAR and FRR

    weights = {}
    for scenario in info:
        FAR = 0
        FRR = 0
        for threshold in info[scenario]:
            far = info[scenario][threshold]["far"]
            frr = info[scenario][threshold]["frr"]
            n_genuine_faces = info[scenario][threshold]["n_genuine_faces"]
            n_impostor_faces = info[scenario][threshold]["n_impostor_faces"]
            #PER OGNI THREASHOLD, FACCIO LA MEAN TRA TUTTE LE FAR E FRR
            #PER OGNI SCENARIO, FACCIO LA WEIGHTED TRA TUTTE LE FAR E FRR
            FAR += far
            FRR += frr
            totalNumberOfFrames = n_genuine_faces + n_impostor_faces
        FAR = FAR/9
        FRR = FRR/9
        weights[scenario] = (FAR, FRR, totalNumberOfFrames)

    totalSum = 0

    FRR = 0
    FAR = 0

    for entry in weights:
        print(entry)
        print(weights[entry])
        totalSum += weights[entry][2]
        FRR += weights[entry][1]*weights[entry][2]
        FAR += weights[entry][0]*weights[entry][2]

    FRR = FRR/totalSum
    FAR = FAR/totalSum
    print("FRR: " + str(FRR))
    print("FAR: " + str(FAR))
    




            
            
            



