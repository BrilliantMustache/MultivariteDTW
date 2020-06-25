from Methods import *

datapath= "/home/xshen5/daniel/TriangleDTW/Data/Multivariate_pickled/"
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

print(">>>>>> Measure prime times.")
MeasurePerformance.MeasurePrimeTimes \
    (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, windows_g[0], Ks_g, Qs_g)

print(">>>>>> Start Z9")
Z9.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)

print(">>>>>> Start X0a")
X0_a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
X0_a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)

print(">>>>>> Start X0")
X0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
X0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, machineRatios)

print(">>>>>> Start X1e")
X1e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
X1e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start X1")
X1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
X1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start X2")
X2.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)
X2.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, K_g)

print(">>>>>> Start X3a")
X3_a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3_a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3")
X3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3ra")
X3ra.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3ra.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3r")
X3r.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3r.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3rsea")
X3rsea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3rsea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3rse")
X3rse.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)
X3rse.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3s")
X3s.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
X3s.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start X3z")
X3z.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
X3z.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z0a")
Z0a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
Z0a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)

print(">>>>>> Start Z0")
Z0.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g)
Z0.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)

print(">>>>>> Start Z1a")
Z1a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start Z1ea")
Z1ea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1ea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start Z1e")
Z1e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start Z1")
Z1.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
Z1.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

print(">>>>>> Start Z3a")
Z3a.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Z3a.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z3ea")
Z3ea.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Z3ea.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z3e")
Z3e.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Z3e.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print(">>>>>> Start Z3")
Z3.dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g, TH_g)
Z3.dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, Ks_g, Qs_g)

print("End")
