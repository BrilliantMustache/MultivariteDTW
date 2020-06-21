from SourceCode.Util import *

# This file implements quantization-based clustering for LB_MV.
# It is the online version, with adaptive cluster numbers.

# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips and coreTime in each individual directory: a text file

def getLB_oneQ_qbox (X, others, qbounds):
    '''
    Get the lower bounds between one query series X and many candidate series in others
    :param X: one series
    :param others: all candidate series
    :param qbounds: the bounding boxes of the query windows
    :return: the lower bounds between X and each candidate series
    '''
    lbs = []
    dim = len(X[0])
    for idy, s2 in enumerate(others):
        LB_sum = 0
        for idy, y in enumerate(s2):
            l=qbounds[idy][0]
            u=qbounds[idy][1]
            temp = math.sqrt(
                sum([(y[idd] - u[idd]) ** 2 if (y[idd] > u[idd]) else (l[idd] - y[idd]) ** 2 if (y[idd] < l[idd]) else 0
                     for idd in range(dim)]))
            LB_sum += temp
        lbs.append(LB_sum)
    return lbs

def DTWDistanceWindowLB_Ordered_Z0 (i, DTWdist, query, references, W):
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
    bounds = []
    for idx in range(ql):
        segment = query[(idx - W if idx - W >= 0 else 0):(idx + W + 1 if idx + W <= ql-1 else ql)]
        l = [min(segment[:, idd]) for idd in range(dim)]
        u = [max(segment[:, idd]) for idd in range(dim)]
        bounds.append([l, u])
    LBs = getLB_oneQ_qbox(query, references, bounds)
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


def dataCollection(datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20]):
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

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

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

            results = [DTWDistanceWindowLB_Ordered_Z0 (ids1, distances,
                                                      query[ids1], reference, windowSize) for ids1 in range(len(query))]
            if findErrors(dataset,maxdim,w,nqueries,nreferences,results,pathUCRResult):
                print('Wrong Results!! Dataset: '+dataset)
                exit()
#            np.save(pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/"
#                    + str(nqueries) + "X" + str(nreferences) + "_Z0_lbs.npy", np.array(LBs_g))
            with open(toppath + str(nqueries) + "X" + str(
                    nreferences) + "_X0" + "_results.txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')
            f.close()

        print(dataset+" Done!"+'\n'+'\n')

    return 0


########################
# collect X0's data
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    allTimes_g = []
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    dataCollection(datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
#    dataProcessing(maxdim_g, nqueries_g, nreferences_g, windows_g)
    print("End")