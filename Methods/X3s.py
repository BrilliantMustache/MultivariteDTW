from Methods.Util import *
from Methods.X3  import findBoundingBoxes
import time


# This file implements X0 followed by X3 for cases that X0 fails but not miserably.
# X3: offline find bounding boxes; online compute the lower bounds when necessary.
# The dataCollection function saves the following:
#     the DTW distances, nearest neighbors, skips, coreTime, X3 invoked times in each individual directory: a text file


def getLB_oneQR_boxR (Q, bboxes):
    '''
    Get the lower bounds between two series, Q and the series with bboxes as its multiple bounding boxes.
    :param Q: A series.
    :param bboxes: the bounding boxes of the other series.
    :return: the lower bound
    '''
    #  X and Y one series, is all references, dim is dimensions, sl_bounds has all the bounding boxes of all reference series
    LB_sum = 0
    dim = len(Q[0])
    for idq, q in enumerate(Q):
        numBoxes = len(bboxes[idq])
        bounds=[]
        for idbox in range(numBoxes):
            l = bboxes[idq][idbox][0]
            u = bboxes[idq][idbox][1]
            temp = math.sqrt(sum([(q[idd]-u[idd]) ** 2 if (q[idd] > u[idd]) else (l[idd]-q[idd])**2
                                    if (q[idd] < l[idd]) else 0 for idd in range(dim)]))
            bounds.append(temp)
        LB_sum+=min(bounds)
    return LB_sum

def getBoundingBoxes(references, w, K=4, Q=2):
    print("Bounding boxes finding Start!")
    start = time.time()
    bboxes = [findBoundingBoxes(np.array(ref), K, w, Q) for ref in references]
    end = time.time()
    setuptime2003cluster_q = end - start
    print("Bounding boxes Done!")
    return bboxes, setuptime2003cluster_q

def DTWDistanceWindowLB_Ordered_X3s_ (queryID, X0LBs, DTWdist, bboxes, K, Q, s1, refs, W, TH=1):
    '''
    Compute the shortest DTW between a query and references series.
    :param queryID: the index number of this query
    :param X0LBs: the X0 lower bounds of the DTW between this query and each reference series
    :param DTWdist: the DTW distances between this query and each reference series (to avoid recomputing
                    the distance in each experiment)
    :param bboxes: the precomputed boounding boxes
    :param K: the maximum number of clusters
    :param Q: the quantization level
    :param s1: the query
    :param refs: the references series
    :param W: the half window size
    :param TH: the triggering threshold of more expensive lower bound calculations
    :return: the DTW distance, the neareast neighbor of this query, the number of DTW distance calculations skipped,
             the number of times the clustering-based method is invoked
    '''
    skip = 0
    cluster_cals = 0
    coretime = 0

    start = time.time()
    LBSortedIndex = np.argsort(X0LBs)
#    LBSortedIndex = sorted(range(len(X0LBs)),key=lambda x: X0LBs[x])
    predId = LBSortedIndex[0]
    end = time.time()
    coretime += (end - start)

    dist = DTW(s1, refs[predId], W)

    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if X0LBs[thisrefid] >= dist:
            skip = len(X0LBs) - x
            break
        elif X0LBs[thisrefid] >= dist - TH*dist:
            c_lb = getLB_oneQR_boxR(s1, bboxes[thisrefid])
            cluster_cals += 1
            if c_lb < dist:
                dist2 = DTWdist[queryID][thisrefid]
                if dist > dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skip = len(X0LBs) - x
                break
        else:
            dist2 = DTWdist[queryID][thisrefid]
            if dist > dist2:
                dist = dist2
                predId = thisrefid
    end = time.time()
    coretime += (end - start)

    return dist, predId, skip, coretime, cluster_cals

def dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2], TH=1):
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

    allsetupTimes = []
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
        # -------------------------------------------------
        if (nqueries * nreferences == 0):  # all series to be used
            qfrac = 0.3
            samplequery = stuff[:int(size * qfrac)]
            samplereference = stuff[int(size * qfrac):]
        # -------------------------------------------------

        print(dataset + ":  " + str(nqueries) + " queries, " + str(nreferences) + " references." +
              " Total dtw: " + str(nqueries * nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w) + "/"
            lb2003 = load_M0LBs(pathUCRResult,dataset,maxdim,w,nqueries,nreferences)
            distanceFileName = pathUCRResult + "" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            if not os.path.exists(distanceFileName):
                distances = [[DTW(s1, s2, w) for s2 in reference] for s1 in query]
                np.save(distanceFileName, np.array(distances))
            else:
                distances = np.load(distanceFileName)
            for K in Ks:
                for Q in Qs:
                    print("K="+str(K)+" Q="+str(Q))
                    bboxes, setuptime = getBoundingBoxes(reference, w, K, Q)
                    results = [DTWDistanceWindowLB_Ordered_X3s_ (ids1, lb2003[ids1], distances, bboxes, K, Q,
                                query[ids1], reference, windowSize) for ids1 in range(len(query))]
                    if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                        print('Wrong Results!! Dataset: ' + dataset)
                        exit()
                    with open(toppath + str(nqueries) + "X" + str(
                            nreferences) + "_X3s_K" + str(K) + "Q" + str(Q) + "_results.txt", 'w') as f:
                        for r in results:
                            f.write(str(r) + '\n')
                    allsetupTimes.append(setuptime)
    np.save(pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
            + "_X3_s_w" + intlist2str(windows) +"K" + intlist2str(Ks)+ "Q" + intlist2str(Qs) + "_setuptimes.npy", allsetupTimes)
    return 0

def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], Ks=[6], Qs=[2],machineRatios=[1,1]):
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
        pathUCRResult + '_AllDataSets/' + 'd' + str(maxdim) + "/" + str(nqueries) + "X" + str(nreferences)
        + "_X0_w" + intlist2str(windows) + "_times.npy")
    X0tLB = setupLBtimes[:, 1]
    tCore = []
    skips = []
    ## -------------------
    NPairs = []
    if nqueries * nreferences == 0:
        actualNQNRs = np.loadtxt(pathUCRResult + '/usabledatasets_nq_nref.txt').reshape((-1, 2))
        for i in range(len(datasets)):
            actualNQ = actualNQNRs[i][0]
            actualNR = actualNQNRs[i][1]
            NPairs.append(actualNQ * actualNR)
    ## -------------------

    for dataset in datasets:
        for K in Ks:
            for Q in Qs:
                results = readResultFile(
                    pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                        nreferences) + "_X3s_K" + str(K) + "Q"+str(Q)+"_results.txt")
                tCore.append(sum(results[:, 3]))
                skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets, -1))
    skips = np.array(skips).reshape((ndatasets, -1))
    tDTW = np.tile(t1dtw[0:ndatasets], (skips.shape[1], 1)).transpose() * ((skips - NPairs) * -1)
    tsum = rother * tCore + rdtw * tDTW
    tsum_min = np.min(tsum, axis=1)
    setting_chosen = np.argmin(tsum,axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
    overhead = rother* (np.array([tCore[i,setting_chosen[i]] for i in range(tCore.shape[0])]) + X0tLB)
    speedups = (rdtw * t1dtw[0:ndatasets] * NPairs) / (rother*X0tLB + tsum_min)
    overheadrate = overhead/(rdtw * t1dtw[0:ndatasets] * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_s_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_s_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_s_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_settingchosen.npy', setting_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_X3_s_w" + str(window) + "K" + intlist2str(Ks) + "Q" + intlist2str(Qs) + '_overheadrate.npy', overheadrate)

    return 0


###############
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    maxdim_g = 5
    nqueries_g = 3
    nreferences_g = 20
    Ks_g = [4, 6, 8]
    Qs_g = [2, 3, 4]
    windows_g = [20]
    TH_g=1
    dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile,datapath, maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g,TH_g)
    dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g,nqueries_g,nreferences_g,windows_g,Ks_g,Qs_g)

    print("End")