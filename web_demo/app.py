import os
import time
import datetime
import logging
import flask
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision import get_image_backend
from PIL import Image

def pil_loader(path):
    return Image.open(path).convert('RGB')
#    with open(path, 'rb') as f:
#        with Image.open(f) as img:
#            return img.convert('RGB')

def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../')
UPLOAD_FOLDER = '/tmp/demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = io.BytesIO(
            urllib.request.urlopen(imageurl).read())
        image = default_loader(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl)


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
        image = default_loader(imagefile)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    string_buf = io.BytesIO()
    image.save(string_buf, format='png')
    data = base64.b64encode(string_buf.getvalue()).decode("utf-8")
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'pretrained_model_file': (
            '{}/model_best.pth.tar'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/lab_ref.csv'.format(REPO_DIRNAME)),
    }


    #default_args['arch'] = 'resnet18'
    #default_args['class_num'] = 1000
    
    def __init__(self, num_classes, arch,
                 class_labels_file, gpu_mode=True, pretrained_model_file=None):
        logging.info('Loading net and associated files...')
        self.net = models.__dict__[arch](num_classes = num_classes)

        if gpu_mode:
            self.net = torch.nn.DataParallel(self.net).cuda()
        
        if os.path.exists(pretrained_model_file):
            checkpoint = torch.load(pretrained_model_file)
            self.net.load_state_dict(checkpoint['state_dict'])

        labels_df = pd.read_csv(class_labels_file)
        self.labels = labels_df.sort_values(by=["id"], ascending=True)['name'].values

        self.loader = default_loader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    def update_model(self, pretrained_model_file=None):
        if pretrained_model_file is not None:
            checkpoint = torch.load(pretrained_model_file)
            self.net.load_state_dict(checkpoint['state_dict'])

    def classify_image(self, image):
        try:
            starttime = time.time()
            input_img = torch.autograd.Variable(self.preprocess(image).unsqueeze(0))
            output = F.softmax(self.net(input_img))
            endtime = time.time()
            
            #score, pred = output.data.max(1)
            scores = output.data.cpu().numpy()[0,:]
            logging.info(str(scores))
            indices = (-scores).argsort()
            logging.info(str(indices))
            use_num = min(5, len(indices))

            res = [(self.labels[idx], scores[idx]) for idx in indices[:use_num]]
            
            logging.info('result: %s', str(res))

            return (True, "meta", res, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


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
    model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch ImageClassification Web Demo')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
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
    args = parser.parse_args()
    


    args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': args.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(arch = args.arch, num_classes = args.num_classes, **ImagenetClassifier.default_args)
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
