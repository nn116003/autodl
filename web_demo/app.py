import os
import time
import datetime
import logging
import flask
from flask import make_response, jsonify
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
#import cStringIO as StringIO
import io
import urllib.request
import exifutil

import base64


from models import ImageClassifier
from web_utils import *


TYPE = "APP" # APP / API

# Obtain the flask app object
app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    if TYPE == "APP":
        return flask.render_template('index.html', has_result=False)
    else:
        return "Hello world."


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = io.BytesIO(
            urllib.request.urlopen(imageurl).read())
        image = default_loader(string_buffer) # 0-255

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        if TYPE == "APP":
            return flask.render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open image from URL.')
            )
        else:
            return make_response(jsonify(ERR_DICT("img_url_err")))

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    if TYPE == "APP":
        return flask.render_template(
            'index.html', has_result=True, result=result, imagesrc=imageurl)
    else:
        if result[0]:
            return make_response(jsonify(result[2]))
        else:
            return make_response(jsonify(ERR_DICT("img_classify_err")))


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = default_loader(imagefile) # 0-255

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        if TYPE == "APP":
            return flask.render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open uploaded image.')
            )
        else:
            return make_response(jsonify(ERR_DICT["img_file_err"]))


    result = app.clf.classify_image(image)

    if TYPE == "APP":
        return flask.render_template(
            'index.html', has_result=True, result=result,
            imagesrc=embed_image_html(image)
        )
    else:
        if result[0]:
            return make_response(jsonify(result[2]))
        else:
            return make_response(jsonify(ERR_DICT("img_classify_err")))






def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch ImageClassification Web Demo')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture ' +
                        ' (default: resnet18)')
    parser.add_argument('-c', '--num_classes', default=1000, type=int, metavar='N',
                    help='number of class (default: 1000)')
    parser.add_argument(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_argument(
        '-p', '--port',
        help="which port to serve content on",
        type=int, default=5000)
    parser.add_argument(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)
    parser.add_argument(
        '-ap', '--api',
        help="api mode",
        action='store_true', default=False)

    args = parser.parse_args()

    if args.api:
        global TYPE
        TYPE = 'API'
    
    ImageClassifier.default_args.update({'gpu_mode': args.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImageClassifier(arch = args.arch, num_classes = args.num_classes, **ImageClassifier.default_args)
#    app.clf.net.forward()

    if args.debug:
        app.run(debug=True, host='0.0.0.0', port=args.port)
    else:
        start_tornado(app, args.port)


if __name__ == '__main__':

    
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
