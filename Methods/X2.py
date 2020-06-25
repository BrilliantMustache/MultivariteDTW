import numpy as np
import pandas as pd
import glob
import math
import pickle as pk
import random as rd
import os

import time
from sklearn.cluster import KMeans
from Methods.Util import *

# This file implements K-Means based LB_MV.
# The dataCollection function saves the following:
#     the lower bounds in each individual directory: an nd array
#     the DTW distances and skips and coreTime in each individual directory: a text file
#     the setup time and total lower bound time of each dataset in one overall file in AllDataSets directory: an nd array


def getLBs (dataset, query, reference, w, dim, K=4):
    nqueries = len(query)
    length=len(query[0])
    nrefs=len(reference)
    windowSize = w if w <= length / 2 else int(length / 2)
    print("W=" + str(windowSize) + '\n')

    print("Starting cluster-2003 ....")
    #  Calculate slices range
    print("Bounding boxes finding Start!")
    start=time.time()
    bboxes = [findBoundingBoxes(np.array(ref), K, windowSize) for ref in reference]
    end=time.time()
    setuptime2003cluster=end-start
    print("Bounding boxes Done!")

    #  Calculate Lower Bounds
    print("Cluster-2003 lower bounds. Start!")
    start=time.time()
    lbs_2003_cluster = [getLB_oneQ (query[ids1], reference, dim, bboxes) for ids1 in range(len(query))]
    end=time.time()
    lbtime2003cluster=end-start
    # np.save(pathUCRResult+"" + dataset + "/d" + str(maxdim_g) + '/w' + str(w) + "/"+
    #         str(nqueries_g)+"X"+str(nreferences_g)+"_X2_K"+str(Ks_g)+"_lbs.npy", lbs_2003_cluster)
    print("Cluster-2003 Done!" + '\n')
    return lbs_2003_cluster, [setuptime2003cluster, lbtime2003cluster]


def loadSkips (datasets, maxdim, windowSizes, nqueries, nrefs, K):
    skips_all = []
    for dataset in datasets:
        for idx, w in enumerate(windowSizes):
            with open(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"+
                              str(nqueries)+"X"+str(nrefs)+ "_X2_K" + str(K) + "_results.txt", 'r') as f:
                temp = f.readlines()
                temps = [l.strip()[1:-1] for l in temp]
                results = [t.split(',') for t in temps]
                skips = [int(r[2]) for r in results]
                skips_all.append(skips)
    return skips_all


#--------------- Cluster-2003 -------------
def findBoundingBoxes(ref, K, W):
    '''
    find the K bounding boxes for each window in ref
    :param ref: a data frame holding a reference series
    :param K: the number of bounding boxes
    :param W: the window size
    :return: a len(ref)*K array with each element [ [dim low ends] [dim high ends]]
    '''
    length = ref.shape[0]
    dims = ref.shape[1]
    allBoxes = []
    for idx in range(length):
        awindow = ref[(idx - W if idx - W >= 0 else 0):(idx + W if idx + W <= length else length)]
        clusterRst = KMeans(n_clusters=K, random_state=0).fit(awindow)
        numberofclusters = len(clusterRst.cluster_centers_)
        if (numberofclusters<K):
            print("less than K clusters found.")
        groups = [[] for i in range(numberofclusters)]
        bboxes = []
        for i in range(len(awindow)):
            clusterID = clusterRst.labels_[i]
            groups[clusterID].append(awindow[i])
        l = []
        u = []
        for g in groups:
            if len(g)>0:
                try:
                    l = [min(np.array(g)[:,idd]) for idd in range(dims)]
                except:
                    print("g is wrong.")
                u = [max(np.array(g)[:,idd]) for idd in range(dims)]
                bboxes.append([l, u])
        for g in groups:
            if len(g)==0:
                bboxes.append([l,u])
        allBoxes.append(bboxes)
    return np.array(allBoxes)

def getLB_oneQ (X, others, dim, sl_bounds):
    #  X is one series, others is all references, dim is dimensions, sl_bounds has all the bounding boxes of all reference series
    lbs = []
    numBoxes = sl_bounds[0][0].shape[0]
    for idy, s2 in enumerate(others):
        temps = []
        LB_sum = 0
        slboundsOneY = sl_bounds[idy]
        for idx, x in enumerate(X):
            oneYbounds=[]
            for idbox in range(numBoxes):
                l = slboundsOneY[idx][idbox][0]
                u = slboundsOneY[idx][idbox][1]
                temp = math.sqrt(sum([(x[idd]-u[idd]) ** 2 if (x[idd] > u[idd]) else (l[idd]-x[idd])**2 if (x[idd] < l[idd]) else 0
                               for idd in range(dim)]))
                oneYbounds.append(temp)
            LB_sum+=min(oneYbounds)
        lbs.append(LB_sum)
    return lbs
#------------------------------------------

def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], K=4):
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

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    # #datasets = ["CharacterTrajectories"]
    #
    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    allTimes=[]
    for idxset, dataset in enumerate(datasets):
        print(dataset+" Start!")
        assert(datasize[idxset]>=nqueries+nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset,nqueries+nreferences)
        size = len(stuff)
        length = stuff[0].shape[0]
        dim = min(stuff[0].shape[1], maxdim)
        print("Size: "+str(size))
        print("Dim: "+str(dim))
        print("Length: "+str(length))
        samplequery = stuff[:nqueries]
        samplereference = stuff[nqueries:nreferences+nqueries]

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            lbs_X2, times = getLBs (dataset, query, reference, w, dim, K)
            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
                    + str(nqueries) + "X" + str(nreferences) + "_X2_K" + str(K) + "_lbs.npy", lbs_X2)
            allTimes.append(times)
            results=get_skips (dataset, maxdim, w, lbs_X2, query, reference)
            if findErrors(dataset,maxdim,w,nqueries,nreferences,results,pathUCRResult):
                print('Wrong Results!! Dataset: '+dataset)
                exit()
            with open(pathUCRResult + dataset + '/' + 'd' + str(maxdim) + '/w'+ str(w) + "/" + str(nqueries) + "X" + str(
                    nreferences) + "_" + "X2_K" + str(K) + "_results" + ".txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')

        print(dataset+" Done!"+'\n'+'\n')

    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/"+ str(nqueries) + "X" + str(nreferences) +
            "_X2_w" + intlist2str(windows) + "K" + str(K) + "_times.npy", allTimes)

def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], K=4, machineRatios=[1,1]):
    datasets = []
    # with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    window = windows[0]
    rdtw = machineRatios[0]
    rother = machineRatios[1]
    t1dtw = loadt1dtw(pathUCRResult, maxdim, window)

#    datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

    ndatasets = len(datasets)

    # compute speedups
    setupLBtimes = np.load(
        pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences) +
        "_X2_w" + intlist2str(windows) + "K" + str(K) + "_times.npy")
    tLB = setupLBtimes[:, 1]
    tCore = []
    skips = []
    totalPairs = nqueries * nreferences
    NPairs = np.array([totalPairs for i in range(ndatasets)])
    for dataset in datasets:
        results = readResultFile(
            pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                nreferences) + "_X2_K" + str(K) + "_results.txt")
        tCore.append(sum(results[:, 3]))
        skips.append(sum(results[:, 2]))
    tCore = np.array(tCore)
    tDTW = t1dtw * (NPairs - np.array(skips))
    speedups = (rdtw*t1dtw * NPairs) / (rdtw*(tLB + tCore) + rother*tDTW)
    overheadrate = rdtw*(tLB + tCore)/(rdtw*t1dtw * NPairs)

    np.save(pathUCRResult+"_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X2_w"+str(window)+'K'+str(K)+'_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X2_w" + str(window) + 'K' + str(K) + '_skips.npy', skips)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X2_w" + str(window) + 'K' + str(K) + '_overheadrate.npy', overheadrate)
    return 0

############################################
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    K_g = 4
    dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g, K_g)

    dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g,nqueries_g,nreferences_g,windows_g, K_g)

    print("End")
