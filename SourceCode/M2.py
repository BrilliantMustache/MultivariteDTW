import numpy as np
import pandas as pd
import glob
import math
import pickle as pk
import random as rd
import os

import time
from sklearn.cluster import KMeans
from SourceCode.Util import *

# This file implements K-Means based LB_MV

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
    #         str(nqueries_g)+"X"+str(nreferences_g)+"_M2K"+str(Ks_g)+"_lbs.npy", lbs_2003_cluster)
    print("Cluster-2003 Done!" + '\n')

    # thistimes = [setuptime2003cluster, lbtime2003cluster]
    #
    # np.save(pathUCRResult+"" + dataset + "/d" + str(maxdim_g) + '/w' + str(w) + "/"+ str(nqueries_g)+"X"+str(nreferences_g)+"_M2K"+str(Ks_g)+"_times.npy", thistimes)
    #
    # allTimes_g.append([setuptime2003cluster, lbtime2003cluster])

    return lbs_2003_cluster, [setuptime2003cluster, lbtime2003cluster]


def loadSkips (datasets, maxdim, windowSizes, nqueries, nrefs, K):
    skips_all = []
    for dataset in datasets:
        for idx, w in enumerate(windowSizes):
            with open(pathUCRResult + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"+
                              str(nqueries)+"X"+str(nrefs)+ "_M2K" + str(K) + "_results.txt", 'r') as f:
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

def dataCollection(datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], K=4):
    datasets=[]
    #with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open("Results/UCR/size.txt",'r') as f:
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

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
            lbs_M2, times = getLBs (dataset, query, reference, w, dim, K)
            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
                    + str(nqueries) + "X" + str(nreferences) + "_M2K" + str(K) + "_lbs.npy", lbs_M2)
            allTimes.append(times)
            results=get_skips (dataset, maxdim, w, lbs_M2, query, reference)
            with open(pathUCRResult + dataset + '/' + 'd' + str(maxdim) + '/w'+ str(w) + "/" + str(nqueries) + "X" + str(
                    nreferences) + "_" + "M2K" + str(K) + "_results" + ".txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')

        print(dataset+" Done!"+'\n'+'\n')

    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/"+ str(nqueries) + "X" + str(nreferences) +
            "_M2w" + intlist2str(windows) + "K" + str(K) + "_times.npy", allTimes)

def dataProcessing(maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], K=4):
    datasets=[]
    #with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open(pathUCRResult+"size.txt",'r') as f:
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

    # # get times
    # M0M2Times  = np.load(pathUCRResult+"" + '/' + str(nqueries_g) + "X" + str(nreferences_g) + "_times_2003_cluster.npy")
    # #  [ dataset1_[2003setupTime 2003LBTime M2setupTime M2LBTime] dataset2_... ]
    # datasetsNum = M0M2Times.shape[0]
    # M2SetupTimes = [ M0M2Times[d][2] for d in range(datasetsNum)]
    # M2LBTimes = [ M0M2Times[d][3] for d in range(datasetsNum)]
    datasetsNum = len(datasets)

    allM2Times = np.load(pathUCRResult+"_AllDataSets/" +'d' + str(maxdim) + '/' + str(nqueries) + "X" +
                         str(nreferences) + "_M2w"+intlist2str(windows) + "_times.npy")
    M2SetupTimes = [ allM2Times[d][0] for d in range(datasetsNum)]
    M2LBTimes = [allM2Times[d][1] for d in range(datasetsNum)]

    # [ [data1_SetupTime data1_LBTime] [data2_SetupTime data2_LBTime] ... ]
    allM0Times = np.load(pathUCRResult+"_AllDataSets/" +'d' + str(maxdim) + '/' + str(nqueries) + "X" +
                         str(nreferences) + "_M0w"+intlist2str(windows) + "_times.npy")
    M0SetupTimes = [ allM0Times[d][0] for d in range(datasetsNum)]
    M0LBTimes = [allM0Times[d][1] for d in range(datasetsNum)]

    M2SetupRatios = [M2SetupTimes[d]/M0LBTimes[d] for d in range(len(datasets))]
    M2LBRatios = [M2LBTimes[d]/M0LBTimes[d] for d in range(len(datasets))]

    # get skips
    M2Skips = loadSkips(datasets, maxdim, windows, nqueries, nreferences, K)

    # save all the data to files
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M2SetupRatios.npy", M2SetupRatios)
    np.save(pathUCRResult+"UsedForPaper/" + str(nqueries) + "X" + str(nreferences) + "M2LBRatios.npy",
            M2LBRatios)
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M2skips.npy", M2Skips)

    print("data saved.")

############################################
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    allTimes_g = []
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    K_g = 4
    dataCollection(datapath,maxdim_g,nqueries_g,nreferences_g,windows_g, K_g)

    dataProcessing(maxdim_g,nqueries_g,nreferences_g,windows_g, K_g)

    print("End")
