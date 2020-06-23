# MultivariteDTW

This is a project containing a set of baseline and novel methods for conducting multivariate DTW.

====== List of Methods ==================
(X: for Xianxia which means offline in Chinese; Z: for Zaixian which means online in Chinese)

X0: LV_MV_offline: min and max are precomputed on references.
Z0: LV_MV_online.

X1: LV_MV_TIPTOP_offline. X0 followed by online TIPTOP.
Z1: LV_MV_TIPTOP_online. Z0 followed by online TIPTOP.

X2: LV_PC_offline_kMeans: Point clustering with k-Means. Done on reference series offline.

X3: LV_PC_offline_quan: point clustering-based X0 with adaptive quantizations. Quantization on reference series offline. No X0 is used.
Xr3: LV_PC_offline_quan with window boxes reuse to reduce space cost.
Xs3: X0 followed by X3. Both use offline setups.
Xz3: LV_PC_mixed_quan: X0 followed by Z3
Z3: LV_PC_online_quan: Z0 followed by point clustering-based Z0 with adaptive quantizations. Done on query series online.

Z4: LV_AD_online: compute all point distances to get the lower bounds.

Z5: NoLB: directly compute all the DTW distances without using lower bounds.

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
            'X3K4Q2': the method is X3, and the maximum number of clusters is 4, the number of quantization levels per
             dimension is at most 2.
            'Z0Z1Z4': the info about Z0, Z1, and Z4
  fileContent: the content of the file
         'lbs': lower bounds
         'results': DTW-NN results. Each row has (DTWdistance, Nearest Neighbor ID, Number of skips)
         'DTWdisances': DTW distances.
  suffix:
         'npy': numbpy file
         'txt': text file
