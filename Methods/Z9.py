from Methods.Util import *

def DTWDistanceWindowLB_Ordered_Z9 (i, DTWdist, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param i: the query ID number
    :param DTWdist: precomputed DTW distances (for fast experiments)
    :param query: the query series
    :param references: a list of reference series
    :param W: half window size
    :return: the DTW distance and the nearest neighbor and the coretime
    '''
    skip = 0

    start = time.time()
    # get bounds of query
    ql = len(query)
    dim = len(query[0])
    bounds = []
    nn = np.argmin(DTWdist[i])
    mindist = DTWdist[i][nn]
    end = time.time()
    coreTime = end - start
#    LBs_g.append(LBs)

    return mindist, nn, 0, coreTime

def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim=5, nqueries=3,
               nreferences=20, windows=[20]):
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

    # # create directories if necessary
    # for datasetName in datasets:
    #     for w in windows:
    #         dir = pathUCRResult+"" + datasetName + "/" + str(w)
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)

    for idxset, dataset in enumerate(datasets):
        print(dataset + " Start!")
        assert (datasize[idxset] >= nqueries + nreferences)
        stuff = loadUCRData_norm_xs(datapath, dataset, nqueries + nreferences)
        size = len(stuff)
        length = stuff[0].shape[0]
        dim = min(stuff[0].shape[1], maxdim)
        print("Size: " + str(size))
        print("Dim: " + str(dim))
        print("Length: " + str(length))
        samplequery = stuff[:nqueries]
        samplereference = stuff[nqueries:nreferences + nqueries]

        print(dataset + ":  " + str(nqueries) + " queries, " + str(nreferences) + " references." +
              " Total dtw: " + str(nqueries * nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w) + "/"
            distanceFileName = pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            #if not os.path.exists(distanceFileName):
            distances = [[DTW(s1, s2, w) for s2 in reference] for s1 in query]
            np.save(distanceFileName, np.array(distances))
            #else:
            #    distances = np.load(distanceFileName)

            results = [DTWDistanceWindowLB_Ordered_Z9 (ids1, distances,
                                                      query[ids1], reference, windowSize) for ids1 in
                       range(len(query))]
            with open(toppath + str(nqueries) + "X" + str(
                    nreferences) + "_Z9" + "_results.txt", 'w') as f:
                for r in results:
                    f.write(str(r) + '\n')
            f.close()

        print(dataset + " Done!" + '\n' + '\n')

    return 0
########################
# collect X0's data
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    allTimes_g = []
    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath,maxdim_g,nqueries_g,nreferences_g,windows_g)
    #dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g)
    print("End")