from SourceCode.Z0 import getLB_oneQ_qbox
from SourceCode.X1 import *

def DTWDistanceWindowLB_Ordered_Z1 (queryID, DTWdist, TH, P, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param TH: the triggering threshold for the expensive filter to take off
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the coretime
    '''
    skips = 0
    p_cals = 0
    coretime = 0

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
    end=time.time()
    coretime += (end - start)

    dist,dxx  = DTWwnd (query, references[predId], W)

    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        if LBs[LBSortedIndex[x]] > dist:
            skips += 1
        elif LBs[LBSortedIndex[x]] >= dist - TH*dist:
            p_lb = tiBounds_top_calP_list_comp(query, references[LBSortedIndex[x]], P, W, dxx)
            p_cals += 1
            if p_lb <= dist:
                dist2 = DTWdist[queryID][LBSortedIndex[x]]
                if dist >= dist2:
                    dist = dist2
                    predId = LBSortedIndex[x]
            else:
                skips += 1
        else:
            dist2 = DTWdist[queryID][LBSortedIndex[x]]
            if dist >= dist2:
                dist = dist2
                predId = LBSortedIndex[x]

    end = time.time()
    coretime += (end - start)
#    LBs_g.append(LBs)

    return dist, predId, skips, coretime, p_cals

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

#    datasets=["ArticularyWordRecognition","AtrialFibrillation"]

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
            distanceFileName = pathUCRResult+"" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            if not os.path.exists(distanceFileName):
                distances = [[DTW(s1, s2, w) for s2 in reference] for s1 in query]
                np.save(distanceFileName, np.array(distances))
            else:
                distances = np.load(distanceFileName)
#            dists = [[DTW(s1, s2, windowSize) for s2 in reference] for s1 in query]
            for TH in THs:
                results = [DTWDistanceWindowLB_Ordered_Z1 (ids1, distances, TH,
                            period_g, query[ids1], reference, windowSize) for ids1 in range(len(query))]
                if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                    print('Wrong Results!! Dataset: ' + dataset)
                    exit()
                with open(toppath+ str(nqueries) + "X" + str(
                    nreferences) + "_Z1TH"+str(TH)+"_results.txt", 'w') as f:
                    for r in results:
                        f.write(str(r)+'\n')
                f.close()
                # with open(toppath+ str(nqueries) + "X" + str(
                #     nreferences) + "_X1TH"+str(TH)+"_times.txt", 'w') as f:
                #     f.write(str(end-start)+'\n')
                #     f.write(str(periodTime)+'\n')
                # f.close()
                # allResults.append(results)
#    np.save(pathUCRResult+"" + '/_AllDataSets/' + "/d"+ str(maxdim) + "/" + str(nqueries)+"X"+str(nreferences)
#            + "_X1"+"w" + intlist2str(windows)+ "TH"+intlist2str(THs) + "_times.npy", allTimes)
    return 0



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
    dataCollection (datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
    #dataProcessing(maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

    print('Done.')