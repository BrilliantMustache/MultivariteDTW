# MultivariteDTW

This is a project containing a set of baseline and novel methods for conducting multivariate DTW.

====== List of Methods ==================
M0: LV_MV
M1: LV_MV_TIPTOP
M2: LV_PC_kMeans: Point clustering with k-Means. Done on reference series offline.
M3: LV_PC_offline: Point clustering with adaptive quantizations. Done on reference series offline.
M4: LV_PC_online: Point clustering with adaptive quantizations. Done on query series online.
M5: LV_AD: compute all point distances to get the lower bounds.
M6: NoLB: directly compute all the DTW distances without using lower bounds.

====== Data File Naming Convention ======
The naming convention used for the output files by the methods:

  Results/UCR/[dataset]/d[dim]/w[windowSize]/[usedSeriesNum]_[methodNameAndSetting]_[fileContent].[suffix]

where
  dataset: the name of the dataset
         '_AllDataSets': all data sets
  dim: the first up to [dim] dimensions of the data are used
         'd0': all dimensions are used
  windowsize: the size of a half window
  usedSeriesNum:
         'allQ[q]': all series used, and the first q fraction is used as queries and the rest as references
         'all': all series used, the partition is the same as specified by the default dataset
         '[m]x[n]': the first m series are used as queries, and the next n series are used as references.
  methodNameAndSetting: the method name following by the used settings.
         Example:
            'M3K4Q2': the method is M3, and the maximum number of clusters is 4, the number of quantization levels per
             dimension is at most 2.
  fileContent: the content of the file
         'lbs': lower bounds
         'results': DTW-NN results. Each row has (DTWdistance, Nearest Neighbor ID, Number of skips)
         'DTWdisances': DTW distances.
  suffix:
         'npy': numbpy file
         'txt': text file
