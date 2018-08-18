import base64
import json

from flask import Flask, request, render_template

app = Flask(__name__)

import numpy as np
from skimage import measure
from skimage import io, morphology, feature
from PIL import Image


@app.route('/<id>')
def index(id):
    return render_template("index.html", id=id)

def save_mask_image(bgname, fname, mname, xy):
    bg = Image.open(bgname)
    bg = bg.convert('L').convert('RGBA')
    mask = Image.new("RGBA", bg.size)
    final = Image.new("RGBA", bg.size)

    source = Image.open(fname)
    mask.paste(source, xy)

    final = Image.alpha_composite(final, bg)
    final = Image.alpha_composite(final, mask)

    final.save(mname)

@app.route('/handle_highlight', methods=['GET', 'POST'])
def handle_highlight():
    if request.method == 'POST':
        f = request.files['file']
        bgname = "data/%s_%s.png"%(request.values["id"], request.values["floor"])
        fname = "data/%s_%s.png"%(request.values["id"], request.values["shopId"])
        mname = "data/m_%s_%s.png"%(request.values["id"], request.values["shopId"])
        f.save(fname)

        xy = (int(request.values["offsetLeft"]), int(request.values["offsetTop"]))
        save_mask_image(bgname, fname, mname, xy)

        dst = io.imread(fname, as_grey=True)
        contours = measure.find_contours(dst, 0.5)
        cords = np.concatenate(contours)

        new_img = measure.subdivide_polygon(cords, degree=2, preserve_ends=True)
        appr_img = measure.approximate_polygon(new_img, tolerance=1)

        return json.dumps(appr_img.tolist(), cls=NumpyEncoder)

@app.route('/save_bg_image', methods=['POST'])
def save_bg_image():
    if request.method == 'POST':
        id = request.values["id"]
        floor = request.values["floor"]
        f = request.files['file']
        fname = "%s_%s.png"%(id, floor)
        full_name = 'data/'+fname
        f.save(full_name)

        img = io.imread(full_name, as_grey=True)
        # 检测canny边缘,得到二值图片
        edgs = feature.canny(img, sigma=3)

        chull = morphology.convex_hull_object(edgs)
        io.imsave('data2/'+fname, chull, as_gray=True)

        contours = measure.find_contours(chull, 0.5)
        cords = np.concatenate(contours)

        new_img = measure.subdivide_polygon(cords, degree=2, preserve_ends=True)
        appr_img = measure.approximate_polygon(new_img, tolerance=1)

        return json.dumps(appr_img.tolist(), cls=NumpyEncoder)

@app.route("/save_shop_info", methods=['POST'])
def save_shop_info():
    if request.method == 'POST':
        fname = "data/%s_%s.json"%(request.values["id"], request.values["shopId"])
        data = request.values["data"]
        with open(fname, "w") as w:
            w.write(data)
        return "ok"

@app.route("/save_building_info", methods=['POST'])
def save_building_info():
    if request.method == 'POST':
        fname = "data2/%s_%sF.json"%(request.values["id"], request.values["floor"])
        data = request.values["data"]
        with open(fname, "w") as w:
            w.write(data)
        return "ok"

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

if __name__ == '__main__':
    app.run(debug=True)
