import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
import numpy as np


def parseFile():
    pathFile = sys.argv[1]
    print("The path file is: " + pathFile)

    scenarioToDiz = {}

    with open(pathFile) as f:
        # read two lines at a time
        for line1, line2 in zip(f, f):
            info = line1.split()
            scenarioName = info[1]
            threshold = info[3]
            if scenarioName not in scenarioToDiz:
                scenarioToDiz[scenarioName] = {}
            scenarioToDiz[scenarioName][threshold] = json.loads(line2)
            # the second line should be read as a json
    # pprint(scenarioToDiz)

    return scenarioToDiz


def FARandFRR(info):
    thresholds = {}
    for scenario in info:
        for threshold in info[scenario]:
            if threshold not in thresholds:
                thresholds[threshold] = []
            far = info[scenario][threshold]["far"]
            frr = info[scenario][threshold]["frr"]
            n_genuine_faces = info[scenario][threshold]["n_genuine_faces"]
            n_impostor_faces = info[scenario][threshold]["n_impostor_faces"]
            thresholds[threshold].append((far, frr, n_genuine_faces, n_impostor_faces))

    pprint(thresholds)

    for threshold in thresholds:
        avg_far = np.mean([x[0] for x in thresholds[threshold]])
        avg_frr = np.mean([x[1] for x in thresholds[threshold]])
        thresholds[threshold] = (avg_far, avg_frr)
    return thresholds


def plotFARandFRR(thresholds):
    # plot the results in the same plot
    plt.figure()
    plt.title("FAR and FRR for different thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("FAR and FRR")
    plt.grid(True)
    plt.plot(thresholds.keys(), [x[0] for x in thresholds.values()], 'r', label="FAR")
    plt.plot(thresholds.keys(), [x[1] for x in thresholds.values()], 'b', label="FRR")
    plt.legend()
    plt.show()


def computeERR(threshold):
    EER = 0
    minDiff = 1
    for threshold in thresholds:
        far = thresholds[threshold][0]
        frr = thresholds[threshold][1]
        diff = abs(far - frr)
        if diff < minDiff:
            minDiff = diff
            EER = threshold
    return EER


if __name__ == "__main__":
    info = parseFile()

    thresholds = FARandFRR(info)
    plotFARandFRR(thresholds)
    pprint(thresholds)  # TODO: scrivere l'output nel report
    # compute the EER
    EER = computeERR(thresholds)
    # print("The EER is: " + str(EER))

    # TODO: plottare la dir? magari la tronchiamo

    # TODO: plottare la FRR e la FAR per ogni scenario

    for scenario in info:
        thresholds = FARandFRR({scenario: info[scenario]})
        plotFARandFRR(thresholds)
        pprint(thresholds)
