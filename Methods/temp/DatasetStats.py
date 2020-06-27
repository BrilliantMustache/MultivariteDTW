from Methods.Util import *

datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
pathUCRResult = "../../Results/UCR/"
datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

datasets = []
# with open("Results/UCR/allDataSetsNames.txt",'r') as f:
with open(datasetsNameFile, 'r') as f:
    for line in f:
        datasets.append(line.strip())
f.close()
datasize = []
# with open("Results/UCR/size.txt",'r') as f:
with open(datasetsSizeFile, 'r') as f:
    for line in f:
        datasize.append(int(line.strip()))
f.close()


print("{0:<30} {1:>10} {2:>10} {3:>10}".format("Name","Size","Length","Dim"))
for idxset, dataset in enumerate(datasets):
    stuff = loadUCRData_norm_xs(datapath, dataset, 1)
    length = stuff[0].shape[0]
    dim = stuff[0].shape[1]
    size = datasize[idxset]
    print("{0:<30} {1:>10} {2:>10} {3:>10}".format(dataset, str(size),str(length),str(dim)))