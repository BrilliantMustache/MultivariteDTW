import glob

import os

import shutil

from Methods import *
import numpy as np

def cleanWorkspace (pathUCRResult, distanceFileDir, datasetsNameFile, dataSizeFiles, maxdim, w):
    usableDataSets = []
    datasizes=[]
    sizes = []
    a = input("Are you sure to clean the workspace? (y/n) ")
    if a!='y':
        print("Please rerun me when you are ready.")
        exit()
    datasets = []
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    with open(dataSizeFiles, 'r') as f:
        for line in f:
            datasizes.append(line.strip())

    for idx, d in enumerate(datasets):
        datasetName = d
        if os.path.isdir(pathUCRResult + "/" + d):
            shutil.rmtree(pathUCRResult + "/" + d)
        newdir = pathUCRResult+'/'+ datasetName +'/d'+str(maxdim)+'/w'+str(w)
        os.makedirs(newdir)
        distanceFile = distanceFileDir+'/w='+str(w)+'/'+datasetName+"_DTWdistances.npy"
        if os.path.exists(distanceFile):
            usableDataSets.append(datasetName)
            newFile = newdir+'/0X0_NoLB_DTWdistances.npy'
            dists = np.load(distanceFile)
            np.save(newFile, dists)
            sizes.append(datasizes[idx])
        else:
            print('Dataset '+datasetName+' has no dtw distances.')

    shutil.rmtree(pathUCRResult+'/_AllDataSets/')
    os.makedirs(pathUCRResult+'/_AllDataSets/d'+str(maxdim))
    return usableDataSets, sizes

#datapath= "/home/xshen5/daniel/TriangleDTW/Data/Multivariate_pickled/"
datapath="/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
pathUCRResult = "../Results/UCR/"
distanceFileDir = "../Results/UCR_DTWDistances/"

#datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
#datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"
#datasetsNameFile = pathUCRResult+"allDataSetsNames_firstTwo.txt"
#datasetsSizeFile = pathUCRResult+"size_firstTwo.txt"
datasetsNameFile = pathUCRResult+"only2ndDataset.txt"
datasetsSizeFile = pathUCRResult+"size_only2ndDataset.txt"

maxdim_g = 5
nqueries_g = 0
nreferences_g = 0
windows_g = [20]
machineRatios = [1, 1]
THs_g = [0.05, 0.1, 0.2]
#THs_g = [0.2, 0.4, 0.6, 0.8]
#Ks_g = [4, 6, 8]
Ks_g = [6]
Qs_g = [2, 3]
#THs_g_Z3 = [0.8, 0.5, 0.3, 0.1]
THs_g_Z3 = [0.1, 0.5]
K_g = 4
period_g = 5
#
# ##### Clean workspace
usableDataSets, sizes = cleanWorkspace(pathUCRResult, distanceFileDir, datasetsNameFile, datasetsSizeFile, maxdim_g, windows_g[0])
datasetsNameFile = pathUCRResult+'usableDatasets.txt'
datasetsSizeFile = pathUCRResult+'size_usableDatasets.txt'
with open(datasetsNameFile,'w') as f:
    [f.write(l+'\n') for l in usableDataSets]
with open(datasetsSizeFile, 'w') as f:
    [f.write(l+'\n') for l in sizes]

# # # ##### Running all methods
# print(">>>>>> Measure prime times.")
# MeasurePerformance.MeasurePrimeTimes \
#     (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, windows_g[0], Ks_g, Qs_g)
# #
# print(">>>>>> Start Z9")
# Z9.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)

#
# print(">>>>>> Start X0a")
# X0a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
# X0a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)

#print(">>>>>> Start X0")
#X0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
#X0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)

#print(">>>>>> Start X3rsea")
#X3rsea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#X3rsea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

#exit()


print(">>>>>> Measure prime times.")
MeasurePerformance.MeasurePrimeTimes \
    (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, windows_g[0], Ks_g, Qs_g)

print(">>>>>> Start Z9")
Z9.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)

#
print(">>>>>> Start X0")
X0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
X0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)
#
#
print(">>>>>> Start X0a")
X0a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
X0a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)
#
# print(">>>>>> Start X1e")
# X1e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
# X1e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
print(">>>>>> Start X1ea")
X1ea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
X1ea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
# print(">>>>>> Start X1")
# X1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
# X1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
# # print(">>>>>> Start X2")
# # X2.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)
# # X2.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)
#
# print(">>>>>> Start X3a")
# X3a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# X3a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3")
# X3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# X3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3ra")
# X3ra.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# X3ra.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3r")
# X3r.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# X3r.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#

print(">>>>>> Start X3rsea")
X3rsea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3rsea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3rse")
# X3rse.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# X3rse.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3s")
# X3s.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
# X3s.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start X3z")
# X3z.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
# X3z.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
print(">>>>>> Start Z0a")
Z0a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
Z0a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)
#
# print(">>>>>> Start Z0")
# Z0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
# Z0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)
#
# print(">>>>>> Start Z1a")
# Z1a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
# Z1a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
print(">>>>>> Start Z1ea")
Z1ea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1ea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
# print(">>>>>> Start Z1e")
# Z1e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
# Z1e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

# print(">>>>>> Start Z1")
# Z1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
# Z1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#
# print(">>>>>> Start Z3a")
# Z3a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# Z3a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z3ea")
Z3ea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_Z3)
Z3ea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, THs_g_Z3)

# print(">>>>>> Start Z3e")
# Z3e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# Z3e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
#
# print(">>>>>> Start Z3")
# Z3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
# Z3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

########## archive the results

print("End")