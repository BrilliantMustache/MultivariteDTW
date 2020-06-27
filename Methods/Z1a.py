from Methods.Z0 import getLB_oneQ_qbox
from Methods.X1 import *

def DTWDistanceWindowLB_Ordered_Z1_a(TH, P, query, references, W):
    '''
    Compute the DTW distance between a query series and a set of reference series.
    :param TH: the triggering threshold for the expensive filter to take off
    :param P: the period length in TIP
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
    LBSortedIndex = np.argsort(LBs)
#    LBSortedIndex = sorted(range(len(LBs)),key=lambda x: LBs[x])
    predId = LBSortedIndex[0]
#    end=time.time()
#    coretime += (end - start)

    dist,dxx  = DTWwnd (query, references[predId], W)

#    start = time.time()
    for x in range(1, len(LBSortedIndex)):
        thisrefid = LBSortedIndex[x]
        if LBs[thisrefid] >= dist:
            skips = len(LBs) - x
            break
        elif LBs[thisrefid] >= dist - TH*dist:
            p_lb = tiBounds_top_calP_list_comp(query, references[thisrefid], P, W, dxx)
            p_cals += 1
            if p_lb < dist:
#                dist2 = DTWdist[queryID][thisrefid]
                dist2 = DTW_a(query, references[thisrefid], W, dist)
                if dist >= dist2:
                    dist = dist2
                    predId = thisrefid
            else:
                skips = len(LBs) -x
                break
        else:
            dist2 = DTW_a(query, references[thisrefid], W, dist)
            if dist > dist2:
                dist = dist2
                predId = thisrefid

    end = time.time()
    coretime += (end - start)
#    LBs_g.append(LBs)

    return dist, predId, skips, coretime, p_cals

def dataCollection (pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1], period_g=5):
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
        # -------------------------------------------------
        if (nqueries * nreferences == 0):  # all series to be used
            qfrac = 0.3
            samplequery = stuff[:int(size * qfrac)]
            samplereference = stuff[int(size * qfrac):]
        # -------------------------------------------------

        print(dataset+":  "+ str(nqueries)+" queries, "+ str(nreferences)+ " references." +
              " Total dtw: "+str(nqueries*nreferences))

        query = [q.values[:, :dim] for q in samplequery]
        reference = [r.values[:, :dim] for r in samplereference]

        for w in windows:
            windowSize = w if w <= length / 2 else int(length / 2)
            toppath = pathUCRResult + dataset + "/d" + str(maxdim) + '/w' + str(w)+"/"
            distanceFileName = pathUCRResult+"" + dataset + '/d' + str(maxdim) + '/w' + str(w) + "/" + \
                               str(nqueries) + "X" + str(nreferences) + "_NoLB_DTWdistances.npy"
            for TH in THs:
                results = [DTWDistanceWindowLB_Ordered_Z1_a(TH, period_g, query[ids1], reference, windowSize) for ids1 in range(len(query))]
                if findErrors(dataset, maxdim, w, nqueries, nreferences, results, pathUCRResult):
                    print('Wrong Results!! Dataset: ' + dataset)
                    exit()
                with open(toppath+ str(nqueries) + "X" + str(
                    nreferences) + "_Z1_a_TH"+str(TH)+"_results.txt", 'w') as f:
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


def dataProcessing(datasetsNameFile, pathUCRResult="../Results/UCR/", maxdim = 5, nqueries = 3, nreferences = 20, windows = [20], THs=[0.1], machineRatios=[1,1]):
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
    t1nd = loadt1nd(pathUCRResult, maxdim, window)

#    datasets = ["ArticularyWordRecognition", "AtrialFibrillation"]

    ndatasets = len(datasets)

    # compute speedups
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
        for TH in THs:
            results = readResultFile(
            pathUCRResult + dataset + '/d' + str(maxdim) + "/w" + str(windows[0]) + "/" + str(nqueries) + "X" + str(
                nreferences) + "_Z1_a_TH" + str(TH) + "_results.txt")
            tCore.append(sum(results[:, 3]))
            skips.append(sum(results[:, 2]))
    tCore = np.array(tCore).reshape((ndatasets,-1))
    skips = np.array(skips).reshape((ndatasets,-1))

    tCorePlus = tCore
#    tDTW = np.tile(t1dtw,(skips.shape[1],1)).transpose() * ((skips-totalPairs)*-1)
    tsum = rother*tCorePlus
    tsum_min = np.min(tsum,axis=1)
    setting_chosen = np.argmin(tsum, axis=1)
    skips_chosen = np.array( [skips[i,setting_chosen[i]] for i in range(skips.shape[0])] )
#    overhead = rother* np.array([tCorePlus[i,setting_chosen[i]] for i in range(tCorePlus.shape[0])])
    speedups = (rdtw*t1dtw[0:ndatasets] * NPairs) / tsum_min
#    overheadrate = overhead/(rdtw*t1dtw * NPairs)

    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z1_a_w" + str(window) + 'TH'+intlist2str(THs)+'_speedups.npy', speedups)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z1_a_w" + str(window) + 'TH' + intlist2str(THs) + '_skipschosen.npy', skips_chosen)
    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
            "_Z1_a_w" + str(window) + 'TH' + intlist2str(THs) + '_settingchosen.npy', setting_chosen)
#    np.save(pathUCRResult + "_AllDataSets/" + 'd' + str(maxdim) + '/' + str(nqueries) + "X" + str(nreferences) +
#            "_Z1_a_w" + str(window) + 'TH' + intlist2str(THs) + '_overheadrate.npy', overheadrate)
    return 0


###############
if __name__ == "__main__":
    datapath= "/Users/xshen/Kids/DanielShen/Research/DTW/Triangle/workshop/TriangleDTW/Data/Multivariate_pickled/"
    pathUCRResult = "../Results/UCR/"
    datasetsNameFile = pathUCRResult+"allDataSetsNames_no_EigenWorms.txt"
    datasetsSizeFile = pathUCRResult+"size_no_EigenWorms.txt"

    maxdim_g = 5
    THs_g = [0.05,0.1,0.2]
    nqueries_g = 3
    nreferences_g = 20
    windows_g = [20]
    period_g = 5
    dataCollection(pathUCRResult, datasetsNameFile, datasetsSizeFile, datapath, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)
    dataProcessing(datasetsNameFile, pathUCRResult, maxdim_g, nqueries_g, nreferences_g, windows_g, THs_g)

    print('Done.')