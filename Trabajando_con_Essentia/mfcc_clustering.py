import json as json
import Extract_MFCCs as xmfccs
import glob
import mean_shift_clustering_diego as msc
import toolz.curried as tzc
import toolz as tz


def mfcc_clustering(dirs, cluster_std, random_state):
    files = list(tz.mapcat(lambda dir_: glob.glob(dir_+"*.wav"), dirs))
    mfccs = xmfccs.extract_all_mfccs(files)

    # these are ordered, i.e. files_[0] belongs to features[0]
    files_ = list(map(lambda x: x["file"], mfccs))
    features = list(map(lambda x: x["mean"], mfccs))

    n_clusters, labels, _, _ = msc.meanShift(
        features,  cluster_std, random_state)
    data = tz.pipe(
        range(len(labels)),
        tzc.map(lambda i: {files_[i]: int(labels[i])}),
        tz.merge)

    return {"data": data, "n_clusters": n_clusters}


# json.dumps(mfcc_clustering(
#     ["/home/diego/sc/overtone/taller-abierto/resources/samples/humedad_8_melodic_split_2020_06_30/"], 1.5, 2)
# )
