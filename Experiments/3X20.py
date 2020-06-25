from Methods import *

datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
pathUCRResult = "../Results/UCR/"
datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

maxdim_g = 5
nqueries_g = 3
nreferences_g = 20
windows_g = [20]
machineRatios = [1, 1]
THs_g = [0.05, 0.1, 0.2]
Ks_g = [4, 6, 8]
Qs_g = [2, 3, 4]
K_g = 4
TH_g = 1
period_g = 5

# print(">>>>>> Measure prime times.")
# MeasurePerformance.MeasurePrimeTimes \
#     (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, windows_g[0], Ks_g, Qs_g)
#
# print(">>>>>> Start X0")
# X0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
# X0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)

#print(">>>>>> Start X1")
#X1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
#X1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

#print(">>>>>> Start X2")
#X2.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)
#X2.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)

print(">>>>>> Start X3")
X3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Xr3")
Xr3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
Xr3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Xs3")
Xs3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Xs3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Xz3")
Xz3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Xz3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z0")
Z0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
Z0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)

print(">>>>>> Start Z1")
Z1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start Z3")
Z3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Z3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print("End")