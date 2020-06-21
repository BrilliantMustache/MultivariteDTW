import numpy as np
import pandas as pd
import glob
import math
import pickle as pk
import random as rd
import os
import time
from SourceCode.Util import *

def tiBounds_top_calP_list_comp(X, Y, W, P, dxx):
    # Same as tiBounds except that the true distances are calculated in every P samples of X
    Xlen = list(X.shape)[0]
    Ylen = list(Y.shape)[0]

    upperBounds = np.zeros([Xlen, W*2+1])
    lowerBounds = np.zeros([Xlen, W*2+1])
    overallLowerBounds = np.zeros(Xlen)

    for t in range(0, Xlen):
        startIdx = 0 if t > W else W - t
        if t % P == 0:
            lw = max(0, t - W)
            tp = min(t + W + 1, Ylen)
            dxyInit = np.array([distance(X[t, :], Y[i, :]) for i in range(lw, tp)])

            upperBounds[t, startIdx:startIdx + tp - lw] = dxyInit
            lowerBounds[t, startIdx:startIdx + tp - lw] = dxyInit
            overallLowerBounds[t] = np.amin(dxyInit)
        else:
            startIdx = 0 if t > W else W - t
            lr = 0 if t < W else t - W
            ur = Ylen - 1 if Ylen - 1 < t + W else t + W
            thisdxx = dxx[t - 1]
            startIdx_lr = startIdx - lr + 1
            t_1 = t - 1
            idx = ur - lr - 1
            if t + W <= Ylen - 1:
                upperBounds[t, startIdx:startIdx + ur - lr] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                               range(lr, ur)]
                lowerBounds[t, startIdx:startIdx + ur - lr] = \
                    [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                     else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                        t_1, startIdx_lr + i]
                     for i in range(lr, ur)]
                # the last y point
                temp = distance(X[t, :], Y[ur, :])
                upperBounds[t, startIdx + idx + 1] = temp
                lowerBounds[t, startIdx + idx + 1] = temp
                overallLowerBounds[t] = np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
            else:
                upperBounds[t, startIdx:startIdx + idx + 2] = [upperBounds[t_1, startIdx_lr + i] + thisdxx for i in
                                                               range(lr, ur + 1)]
                lowerBounds[t, startIdx:startIdx + idx + 2] = \
                    [lowerBounds[t_1, startIdx_lr + i] - thisdxx if lowerBounds[t_1, startIdx_lr + i] > thisdxx
                     else 0 if thisdxx < upperBounds[t_1, startIdx_lr + i] else thisdxx - upperBounds[
                        t_1, startIdx_lr + i]
                     for i in range(lr, ur + 1)]
                overallLowerBounds[t] = np.amin(lowerBounds[t, startIdx:startIdx + idx + 2])
    #------------
    return sum(overallLowerBounds)


def DTWDistanceWindowLB_Ordered_TIPX2003(queryID, LBs, DTWdist, TH, P, s1, refs, W):
    '''
    Compute the shortest DTW between a query and references series.
    :param queryID: the index number of this query
    :param LBs: the lower bounds of the DTW between this query and each reference series
    :param DTWdist: the DTW distances between this query and each reference series (to avoid recomputing the distance in each experiment)
    :param TH: the threshold triggering the use of TI based lower bound computations
    :param P: the period length used in the setup of the TI based method
    :param s1: the query
    :param refs: the references series
    :param W: the half window size
    :return: the DTW distance, the neareast neighbor of this query, the number of DTW distance calculations skipped, the number of times the TI method is invoked
    '''
    skip = 0
    p_cals = 0
    global periodTime_g

    dxx = calNeighborDistances(s1)
    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTWdist[queryID][predId]

    for x in range(1, len(LBSortedIndex)):
        if LBs[LBSortedIndex[x]] > dist:
            skip += 1
        elif LBs[LBSortedIndex[x]] >= dist - TH*dist:
            startTm = time.time()
            p_lb = tiBounds_top_calP_list_comp(s1, refs[LBSortedIndex[x]], P, W, dxx)
            p_cals += 1
            if p_lb <= dist:
                dist2 = DTWdist[queryID][LBSortedIndex[x]]
                if dist >= dist2:
                    dist = dist2
                    predId = LBSortedIndex[x]
            else:
                skip += 1
            endTm = time.time()
            periodTime_g +=(endTm-startTm)
        else:
            try:
                dist2 = DTWdist[queryID][LBSortedIndex[x]]
            except:
                print('Wrong.')
            if dist >= dist2:
                dist = dist2
                predId = LBSortedIndex[x]

    return dist, predId, skip, p_cals


def load_M0LBs(dataset, w):
    lb_2003 = np.load(pathUCRResult+dataset+"/d"+ str(maxdim_g) +"/w"+ str(w) + '/' +
                      str(nqueries_g) + "X" + str(nreferences_g) +"_M0_lbs.npy")
    return lb_2003

def loadSkips (datasets, maxdim, windowSizes, nqueries, nrefs, THs):
    skips_all = []
    for dataset in datasets:
        for idx, w in enumerate(windowSizes):
            skips_temp = []
            for TH in THs:
                with open(pathUCRResult + dataset + '/d' + str(maxdim) + '/w'+ str(w) + "/"+str(nqueries)+
                                  "X"+str(nrefs)+ "_M1TH"+str(TH)+"_results.txt", 'r') as f:
                    temp = f.readlines()
                    temps = [l.strip()[1:-1] for l in temp]
                    results = [t.split(',') for t in temps]
                    skips = [int(r[2]) for r in results]
                    skips_temp.append(sum(skips))
            skips_all.append(skips_temp)
    return skips_all

def dataCollection (datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1]):
    datasets=[]
    with open(pathUCRResult+"allDataSetsNames_no_EigenWorms.txt", 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    with open(pathUCRResult+"size_no_EigenWorms.txt",'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()

    allTimes=[]
    for idxset, dataset in enumerate(datasets):
        print(dataset+" Start!")
        assert(datasize[idxset]>=nqueries+nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset, nqueries+nreferences)
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
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w)+"/"
            lb2003 = load_M0LBs(dataset, w)
            distanceFileName = pathUCRResult+"" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            if not os.path.exists(distanceFileName):
                distances = [[DTW(s1, s2, w) for s2 in reference] for s1 in query]
                np.save(distanceFileName, np.array(distances))
            else:
                distances = np.load(distanceFileName)
#            dists = [[DTW(s1, s2, windowSize) for s2 in reference] for s1 in query]
            for TH in THs:
                start = time.time()
                periodTime_g = 0
                results = [DTWDistanceWindowLB_Ordered_TIPX2003(ids1, lb2003[ids1], distances, TH,
                            period_g, query[ids1], reference, windowSize) for ids1 in range(len(query))]
                end = time.time()
                with open(toppath+ str(nqueries) + "X" + str(
                    nreferences) + "_M1TH"+str(TH)+"_results.txt", 'w') as f:
                    for r in results:
                        f.write(str(r)+'\n')
                f.close()
                # with open(toppath+ str(nqueries) + "X" + str(
                #     nreferences) + "_M1TH"+str(TH)+"_times.txt", 'w') as f:
                #     f.write(str(end-start)+'\n')
                #     f.write(str(periodTime)+'\n')
                # f.close()
                # allResults.append(results)
                allTimes.append([(end-start), periodTime_g])
    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
            + "_M1"+"w" + intlist2str(windows)+ "TH"+intlist2str(THs) + "_times.npy", allTimes)
    return 0


def dataProcessing(maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1]):
    datasets=[]
    windowSize = windows[0]
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

    # get times
    allM1Times  = np.load(pathUCRResult+"_AllDataSets/" + 'd'+str(maxdim)+'/'+str(nqueries)+"X"+
                          str(nreferences)+"_M1w"+intlist2str(windows)+"TH"+intlist2str(THs)+"_times.npy")
    #  [ dataset1[ [setupTime LBTime] ... ] ... ]
    allM1Times = allM1Times.reshape((-1, 2*len(THs)))
    datasetsNum = allM1Times.shape[0]
    M1Settings = int(allM1Times.shape[1]/2)
    M1ExtraLBTimes = [ [allM1Times[d][s*2] for s in range(M1Settings)] for d in range(datasetsNum)]

    # [ [data1_SetupTime data1_LBTime] [data2_SetupTime data2_LBTime] ... ]
    allM0Times = np.load(pathUCRResult+"_AllDataSets/" + 'd'+str(maxdim) + "/"+ str(nqueries) +
                         "X" + str(nreferences) + "_M0w"+intlist2str(windows)+"_times.npy")
    M0LBTimes = [allM0Times[d][1] for d in range(datasetsNum)]

    M1LBRatios = [ [1+M1ExtraLBTimes[d][s]/M0LBTimes[d] for s in range(M1Settings)] for d in range(len(datasets))]

    # get skips
    skips_all = loadSkips(datasets, maxdim, windows, nqueries, nreferences, THs)
    M1Skips = np.array(skips_all)
    # M1Skips = np.array(skips_all)
    #
    # # M1Results: [ TH1*DataSet1_query1_[dist, predId, skip, p_cals] ... ]
    # M1Results = np.load(pathUCRResult+"_AllDataSets/" + 'd'+str(maxdim) + "/"+ str(nqueries)+"X"+str(nreferences)
    #                     +"_M1w"+intlist2str(windows)+"TH"+intlist2str(THs)+"_Results.npy")
    # i=0
    # M1Skips=np.zeros((len(THs)*len(datasets)))
    # for t in range(len(THs)*len(datasets)):
    #     for d in range(nqueries):
    #         M1Skips[t] += M1Results[t,d,2]
    # M1Skips=M1Skips.reshape((len(THs),-1)).transpose()

    # save all the data to files
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M1LBRatios.npy", M1LBRatios)
    np.save(pathUCRResult+"UsedForPaper/"+str(nqueries) + "X" + str(nreferences) + "M1skips.npy", M1Skips)

    print("data saved.")


###############
if __name__ == '__main__':
    # Main Entry Point
    pathUCRResult = "../Results/UCR/"
    datapath = "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    maxdim_g = 5
    THs_g = [0.05,0.1,0.2]
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    period_g = 5
    periodTime_g=0
    allTimes_g=[]
    dataCollection (datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
    dataProcessing(maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

    print('Done.')