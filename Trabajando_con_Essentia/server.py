import melody_segmentation_study_version as meloseg
import mfcc_clustering as mfccc
from flask import Flask, json, request

api = Flask(__name__)


@api.route('/melody-segmentation', methods=['post'])
def melody_segmentation():
    """Processes a file into "melodic" segments. 
    Query params are out_dir and filename. Both are paths.
    Returns an array of objects with path of the processed segment (for a wav file),
    and the start and end samples of the original file (from which the new file was made)"""
    print("received req", request.form)
    out_dir = request.form['out_dir']
    filename = request.form['filename']
    opts = json.loads(request.form['opts'])
    print(opts)
    files = meloseg.process_file(out_dir, filename, opts)
    print(files)
    return json.dumps(files)


@api.route('/mfcc-clustering', methods=['post'])
def mfcc_clustering():
    """Clusterize audiofiles with mfcc.
    Receives a `dirs` param which should be an array of directories (paths should end with "/")
    Also receives a `cluster_std` which should be a float and a random_state which should be an int"""
    print(request.form)
    dirs = request.form.getlist('dirs')
    print(dirs)
    cluster_std = float(request.form['cluster_std'])
    random_state = int(request.form['random_state'])
    response = mfccc.mfcc_clustering(dirs, cluster_std, random_state)
    print(response)
    return json.dumps(response)


if __name__ == '__main__':
    api.run()
