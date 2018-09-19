import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import logging
import os
import pandas as pd
import time

from web_utils import REPO_DIRNAME, default_loader

class ImageClassifier(object):
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

        self.net.eval()
            
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

    def classify_image(self, image):# image PIL.Image 0-255
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

            res = [(self.labels[idx], str(scores[idx])) for idx in indices[:use_num]]
            
            logging.info('result: %s', str(res))

            return (True, "meta", res, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

