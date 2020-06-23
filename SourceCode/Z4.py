from SourceCode.Util import *

# This file implements quantization-based clustering for LB_MV.
# It is the online version, with adaptive cluster numbers.

# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips and coreTime in each individual directory: a text file

def getLB_oneQ_qbox (X, others, W):
    '''
    Get the lower bounds between one query series X and many candidate series in others
    :param X: one series
    :param others: all candidate series
    :param W: the query windows
    :return: the lower bounds between X and each candidate series
    '''
    lbs = []
    for s2 in others:
        LB_sum = 0
        for idx in range(len(X)):
            x = X[idx]
            LB_sum += min([distance(x, s2[i]) for i in
                           range(idx - W if idx - W >= 0 else 0, idx + W if idx + W <= len(s2) - 1 else len(s2) - 1)])
        lbs.append(LB_sum)
    return lbs

def DTWDistanceWindowLB_Ordered_Z4 (i, DTWdist, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the coretime
    '''
    skip = 0

    start = time.time()
    # get bounds of query
    ql = len(query)
    dim = len(query[0])
    LBs = getLB_oneQ_qbox(query, references, W)
    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
    dist = DTWdist[i][predId]
    for x in range(1,len(LBSortedIndex)):
        if dist>LBs[LBSortedIndex[x]]:
#           Use saved DTW distances from baseline for quick experiment
            dist2 = DTWdist[i][LBSortedIndex[x]]
            if dist>=dist2:
                dist = dist2
                predId = LBSortedIndex[x]
        else:
            skip = skip + 1
    end = time.time()
    coreTime = end - start
#    LBs_g.append(LBs)

    return dist, predId, skip, coreTime


def dataCollection(datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20]):
    datasets=[]
    #with open("Results/UCR/allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    datasize=[]
    #with open("Results/UCR/size.txt",'r') as f:
    with open(datasetsSizeFile,'r') as f:
        for line in f:
            datasize.append(int(line.strip()))
    f.close()
    datasets=["ArticularyWordRecognition","AtrialFibrillation"]
    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

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
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w) + "/"
            distanceFileName = pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            if not os.path.exists(distanceFileName):
                distances = [[DTW(s1, s2, w) for s2 in reference] for s1 in query]
                np.save(distanceFileName, np.array(distances))
            else:
                distances = np.load(distanceFileName)

            results = [DTWDistanceWindowLB_Ordered_Z4(ids1, distances,
                                                      query[ids1], reference, windowSize) for ids1 in range(len(query))]
            if findErrors(dataset,maxdim,w,nqueries,nreferences,results,pathUCRResult):
                print('Wrong Results!! Dataset: '+dataset)
                exit()
#            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
#                    + str(nqueries) + "X" + str(nreferences) + "_Z4_lbs.npy", np.array(LBs_g))
            with open(toppath + str(nqueries) + "X" + str(
                    nreferences) + "_Z4" + "_results.txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')
            f.close()

        print(dataset+" Done!"+'\n'+'\n')

    return 0



def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], machineRatios=[1,1]):
    '''
    Process the data to get the speedups. Currently, only deals with the first element in windows.
    :param datasetsNameFile:
    :param pathUCRResult:
    :param maxdim:
    :param nqueries:
    :param nreferences:
    :param windows:
    :param machineRatios: Used for cross-machine performance estimation. [r1, r2].
                          r1: tDTW(new machine)/tDTW(this machine);
                          r2: tM0LB(new machine)/tM0LB(this machine), taken as the ratio for all other times.
    :return: 0
    '''
    datasets=[]
    #with open(pathUCRResult+"allDataSetsNames.txt",'r') as f:
    with open(datasetsNameFile, 'r') as f:
        for line in f:
            datasets.append(line.strip())
    f.close()
    window = windows[0]
    rdtw = machineRatios[0]
    rother = machineRatios[1]
    t1dtw = loadt1dtw(pathUCRResult, maxdim, window)

    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

    ndatasets = len(datasets)

    # compute speedups
    tCore = []
    skips = []
    totalPairs = nqueries*nreferences
    NPairs = np.array([totalPairs for i in range(ndatasets)])
    for dataset in datasets:
        results = readResultFile(pathUCRResult + dataset + '/d' + str(maxdim) + "/w"+ str(windows[0]) + "/" + str(nqueries) + "X" + str(nreferences)
            + "_Z4" + "_results.txt")
        tCore.append(sum(results[:,3]))
        skips.append(sum(results[:,2]))
    tCore = np.array(tCore)
    tDTW = t1dtw*(NPairs - np.array(skips))
    speedups = (rdtw*t1dtw*NPairs)/(rother*tCore+rdtw*tDTW)
    overheadrate = (rother*tCore)/(rdtw*t1dtw*NPairs)

    np.save(pathUCRResult+"_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z4w"+str(window)+'_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z4w" + str(window) + '_skips.npy', skips)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z4w" + str(window) + '_overheadrate.npy', overheadrate)
    return 0

########################
# collect X0's data
if __name__ == "__main__":
    datapath= "/Users/Desktop/PycharmProjects/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    allTimes_g = []
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    dataCollection(datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
    dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)
    print("End")