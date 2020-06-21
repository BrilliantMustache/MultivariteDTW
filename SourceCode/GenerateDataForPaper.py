import numpy as np
import pandas as pd
import glob
import os

import scipy


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

maxdim_g = 5
nqueries_g = 3
nreferences_g = 20
windows_g = [20]
allTimes_g = []

datasets = []
with open("Results/UCR/allDataSetsNames_no_EigenWorms.txt", 'r') as f:
    for line in f:
        datasets.append(line.strip())
f.close()

# read in the basis times. Skip Eigenworms dataset.
basisTimes = np.load("Results/UCR/UsedForPaper/"+"basisTime_all30.npy")
basisTimes_1 = basisTimes[0:7,1:3]
basisTimes_2 = basisTimes[8:30,1:3]
basisTimes_a = []
for x in basisTimes_1:
    basisTimes_a.append([float(x1) for x1 in x])
for x in basisTimes_2:
    basisTimes_a.append([float(x1) for x1 in x])
basisTimes_a_array = np.array(basisTimes_a)
np.save("Results/UCR/UsedForPaper/basisTimes.npy", basisTimes_a_array)

basisTimes = np.load("Results/UCR/UsedForPaper/"+"basisTimes.npy")
DTWUnitTime = basisTimes[:,1]
ndatasets = len(basisTimes)
totalPairs = nqueries_g*nreferences_g
NPairs = np.array([totalPairs for i in range(ndatasets)])
baseTotalTime = NPairs*DTWUnitTime
#baseTotalTime = np.array([totalPairs*unitTime for unitTime in DTWUnitTime])

# get the columns of M0
M0LBTime = np.array(basisTimes[:,0])
M0Skips = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M0skips.npy"))
M0DTWTime = np.array([(NPairs[i]-M0Skips[i])*DTWUnitTime[i] for i in range(ndatasets)])
M0TotalTime = M0LBTime+M0DTWTime
M0LBRate = M0LBTime/(totalPairs*DTWUnitTime)
#M0Speedup = np.array([baseTotalTime[i]/M0TotalTime[i] for i in range(ndatasets)])
M0Speedup = baseTotalTime/M0TotalTime


# get the columns of M1
# M1 setting: 3 thresholds
M1LBRatios = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M1LBRatios.npy"))
M1Skips = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M1skips.npy"))
M1settings = M1LBRatios.shape[1]
M1LBTime=np.empty((ndatasets,M1settings))
M1DTWTime=np.empty((ndatasets,M1settings))
M1LBRate=np.empty((ndatasets,M1settings))
for i in range(M1settings):
    M1LBTime[:,i] = M0LBTime*M1LBRatios[:,i]
    M1DTWTime[:,i] = (NPairs-M1Skips[:,i])*DTWUnitTime
    M1LBRate[:, i] = M1LBTime[:,i]/(totalPairs*DTWUnitTime)
M1TotalTime = M1LBTime+M1DTWTime
M1Speedup=np.empty((ndatasets,M1settings))
for i in range(M1settings):
    M1Speedup[:,i] = baseTotalTime/M1TotalTime[:,i]
M1BestSetting = np.argmax(M1Speedup,axis=1)
M1LBRate_b = np.empty(ndatasets)
M1Speedup_b = np.empty(ndatasets)
M1Skips_b = np.empty(ndatasets)
for i in range(ndatasets):
    bestIdx = M1BestSetting[i]
    M1LBRate_b[i] = M1LBRate[i,bestIdx]
    M1Speedup_b[i] = M1Speedup[i,bestIdx]
    M1Skips_b[i] = M1Skips[i,bestIdx]


# get the columns of M3
# M3 settings: cluster number K=[4,6,8];  quantization level per dimeension Q=[2,3,4].
M3SetupRatios = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M3SetupRatios.npy"))
M3LBRatios = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M3LBRatios.npy"))
M3Skips = np.array(np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M3skips.npy"))
M3settings = M3SetupRatios.shape[1]
M3LBTime=np.empty((ndatasets,M3settings))
M3DTWTime=np.empty((ndatasets,M3settings))
M3LBRate=np.empty((ndatasets,M3settings))
for i in range(M3settings):
    M3LBTime[:,i] = M0LBTime*M3LBRatios[:,i]
    M3DTWTime[:,i] = (NPairs-M3Skips[:,i])*DTWUnitTime
    M3LBRate[:, i] = M3LBTime[:,i]/(totalPairs*DTWUnitTime)
M3TotalTime = M3LBTime+M3DTWTime
M3Speedup=np.empty((ndatasets,M3settings))
for i in range(M3settings):
    M3Speedup[:,i] = baseTotalTime/M3TotalTime[:,i]
M3BestSetting = np.argmax(M3Speedup,axis=1)
M3LBRate_b = np.empty(ndatasets)
M3Speedup_b = np.empty(ndatasets)
M3Skips_b = np.empty(ndatasets)
for i in range(ndatasets):
    bestIdx = M3BestSetting[i]
    M3LBRate_b[i] = M3LBRate[i,bestIdx]
    M3Speedup_b[i] = M3Speedup[i,bestIdx]
    M3Skips_b[i] = M3Skips[i,bestIdx]

# derive the results of selective deployment of M3: use it only when M0 fails
sM3LBRate = [M3LBRate_b[i]-(M3LBRate_b[i]-M0LBRate[i])*(M3Skips_b[i]-M0Skips[i])/M3Skips_b[i] if M3Skips_b[i]>0 else M3LBRate_b[i] for i in range(M3Skips_b.shape[0])]
sM3Speedup = baseTotalTime/(baseTotalTime/M3Speedup_b-M3LBRate_b*baseTotalTime+sM3LBRate*baseTotalTime)
sM3Skips = M3Skips_b

# get the columns of M2
M2SetupRatios = np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M2SetupRatios.npy")
M2LBRatios = np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M2LBRatios.npy")

M2Skips = np.load("Results/UCR/UsedForPaper/" + str(nqueries_g) + "X" + str(nreferences_g) + "M2skips.npy")
M2LBTime = M2LBRatios*M0LBTime
M2DTWTime = (NPairs-M2Skips)*DTWUnitTime
M2Speedup = baseTotalTime/(M2LBTime+M2DTWTime)
M2LBRate = M2LBTime/baseTotalTime

# Write the results out
table=np.array([M0Speedup, M1Speedup_b, M3Speedup_b, sM3Speedup, M2Speedup, M0Skips/totalPairs,
                M1Skips_b/totalPairs, M3Skips_b/totalPairs, sM3Skips/totalPairs , M2Skips/totalPairs,
                M0LBRate, M1LBRate_b, M3LBRate_b, sM3LBRate, M2LBRate]).transpose()
averageRow = [geo_mean(table[:,i]) for i in range(table.shape[1])]
methods=5
methodnames=['M0', 'M1', 'M2', 'sM2', 'kM2']
with open("Results/UCR/UsedForPaper/OverallPerformanceTable.txt", 'w') as f:
    boldnum = '{{\\bf {0:.2f}}}\t&'
    f.write("""\\begin{table*}
    \\centering
    \\caption{Overall Performance Comparison}
    \\label{tab:overall}
    """)
    f.write('\\begin{{tabular}}{{|r||*{{{0}}}{{c|}}c||*{{{0}}}{{c|}}c||*{{{0}}}{{c|}}c|}}\\hline\n'.format(methods-1))
    f.write("Dataset & \\multicolumn{{{0}}}{{c||}}{{Speedups}} & \\multicolumn{{{0}}}{{c||}}{{Skips}} & \\multicolumn{{{0}}}{{c|}}{{LB Overhead}}\\\\\\cline{{2-{1}}}\n".format(methods,methods*3+1))
    for i in [1,2,3]:
        for m in methodnames:
            f.write('& {0} '.format(m))
    f.write('\\\\\\hline\n')
    for idx, dataset in enumerate(datasets):
        f.write(dataset+'\t&')
        f.write('{0:.2f}\t&'.format(M0Speedup[idx]))

        if M1Speedup_b[idx]>=M0Speedup[idx]:
            f.write(boldnum.format(M1Speedup_b[idx]))
        else:
            f.write('{0:.2f}\t&'.format(M1Speedup_b[idx]))

        if M3Speedup_b[idx]>=M0Speedup[idx]:
            f.write(boldnum.format(M3Speedup_b[idx]))
        else:
            f.write('{0:.2f}\t&'.format(M3Speedup_b[idx]))

        if sM3Speedup[idx]>=M0Speedup[idx]:
            f.write(boldnum.format(sM3Speedup[idx]))
        else:
            f.write('{0:.2f}\t&'.format(sM3Speedup[idx]))

        if M2Speedup[idx]>=M0Speedup[idx]:
            f.write(boldnum.format(M2Speedup[idx]))
        else:
            f.write('{0:.2f}\t&'.format(M2Speedup[idx]))

        for e in table[idx,methods:-1]:
            f.write('{0:.2f}\t&'.format(e))
        f.write('{0:.2f}\\\\ \\hline\n'.format(table[idx,-1]))
    f.write('{{\\bf Average}}\t&')
    for e in range(len(averageRow)-1):
        f.write('{0:.2f}\t&'.format(averageRow[e]))
    f.write('{0:.2f}\\\\ \hline\n'.format(averageRow[-1]))
    f.write('\end{tabular}\n')
    f.write('\end{table*}\n')
    f.close()
#np.savetxt("Results/UCR/UsedForPaper/OverallPerformanceTable.txt", table, fmt=["%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f","%3.2f"], delimiter=' & ')


print("pause here.")
