import melody_segmentation_study_version as meloseg
from flask import Flask, json, request

api = Flask(__name__)


@api.route('/melody-segmentation', methods=['post'])
def melody_segmentation():
    """Processes a file into "melodic" segments. 
    Query params are out_dir and filename. Both are paths.
    Returns an array of objects with path of the processed segment (for a wav file),
    and the start and end samples of the original file (from which the new file was made)"""
    out_dir = request.form['out_dir']
    filename = request.form['filename']
    files = meloseg.process_file(out_dir, filename)
    return json.dumps(files)


if __name__ == '__main__':
    api.run()
