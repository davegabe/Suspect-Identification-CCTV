import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
import numpy as np


def parseFile():
    """
    Read the file and return a dictionary with the following structure:
    {
        scenarioName: {
            threshold: {
                "far": far,
                "frr": frr,
                "n_genuine_faces": n_genuine_faces,
                "n_impostor_faces": n_impostor_faces
                ...
            }
        }
    }
    """
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

def computeDIR(info):
    """
    Compute the average DIR for each rank.

    Args:
        info (dict): the dictionary returned by parseFile()

    Returns:
        dict: a dictionary with the following structure:
        {
            threshold: {
                rank: avg_dir
            }
        }
    """
    dir_values: dict[str, dict[str, list[(float, int)]]] = {}
    avg_dir: dict[str, dict[str, float]] = {}
    # for each scenario
    for scenario in info:
        # for each threshold
        for threshold in info[scenario]:
            # for each rank in dir of the current threshold
            for rank in info[scenario][threshold]["dir"]:
                if threshold not in dir_values:
                    dir_values[threshold] = {}
                    avg_dir[threshold] = {}
                if rank not in dir_values[threshold]:
                    dir_values[threshold][rank] = []
                dir_values[threshold][rank].append((info[scenario][threshold]["dir"][rank], info[scenario][threshold]["n_genuine_faces"]))
    # compute the average
    for threshold in dir_values:
        for rank in dir_values[threshold]:
            avg_dir[threshold][rank] = sum(map(lambda x: x[0]*x[1], dir_values[threshold][rank])) /  sum(map(lambda x: x[1], dir_values[threshold][rank])) # avg = sum(a * weights) / sum(weights)
    return avg_dir    


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


def plotFARandFRR(thresholds, scenario=""):
    """
    Plot the FAR and FRR curves for the given thresholds.

    Args:
        thresholds (dict): the dictionary returned by FARandFRR()
        scenario (str, optional): the scenario name. Defaults to "".
    """
    # plot the results in the same plot
    plt.figure()
    if scenario == "":
        plt.title("FAR and FRR")
    else:
        plt.title("FAR and FRR for scenario " + scenario)
        
    plt.xlabel("Threshold")
    plt.ylabel("FAR and FRR")
    # set the title of the window
    plt.grid(True)
    plt.plot(thresholds.keys(), [x[0] for x in thresholds.values()], 'r', label="FAR")
    plt.plot(thresholds.keys(), [x[1] for x in thresholds.values()], 'b', label="FRR")
    plt.legend()
    # plt.show()
    # highlight the intesection point of the two curves
    EER = computeERR(thresholds)
    plt.plot(EER, thresholds[EER][0], 'ro')

    #save the plot in ./plots
    if scenario == "":
        plt.savefig("./plots/FARandFRR.png")
    else:
        plt.savefig("./plots/FARandFRR_" + scenario + ".png")

def plotDIR(info):
    avg_dir = computeDIR(info)
    print("The average DIR is: " + str(avg_dir))
    plt.figure()
    plt.title("CMC") # da verificare, non so come si chiama per open set
        
    plt.xlabel("Rank")
    plt.ylabel("DIR")
    # set the title of the window
    plt.grid(True)
    for threshold in avg_dir:
        if threshold in ["0.3"]:
            plt.plot(avg_dir[threshold].keys(), avg_dir[threshold].values(), label="DIR for threshold " + threshold)
    plt.legend()
    # save the plot in ./plots
    plt.show()
    plt.savefig("./plots/CMC.png")




if __name__ == "__main__":
    info = parseFile()

    # compute the FAR and FRR
    thresholds = FARandFRR(info)
    plotFARandFRR(thresholds)
    pprint(thresholds)  # TODO: scrivere l'output nel report

    # compute the EER
    EER = computeERR(thresholds)
    # print("The EER is: " + str(EER))

    # compute the DIR
    plotDIR(info)

    # for scenario in info:
    #     thresholds = FARandFRR({scenario: info[scenario]})
    #     plotFARandFRR(thresholds, scenario)
    #     pprint(thresholds)

    #compute the CMC curve
    
