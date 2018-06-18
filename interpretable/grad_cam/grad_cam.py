import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from scipy import interpolate
import cv2

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class FeatureExtractor():
    """
    return features, store grads
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        # register hook
        # enumerate feature maps
        # return scores for classes
        pass

class FEResNet(FeatureExtractor):
    def __call__(self, x):
        outputs = [] # feature maps
        self.gradients = []
        for name, module in self.model._modules.items():
            print(name)
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient) # call save_gradient(grad) when backprop is called
                outputs += [x]
        return outputs, x

class FEVGG(FeatureExtractor):
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.features._modules.items():
            print(name)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, self.model.classifier(x.view(x.size(0), -1))

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

    
class GradCam(object):
    def __init__(self, model, target_layer_names):
        model_type = str(type(model))
        if "DataParallel" in model_type:
            model_type = str(type(model.module))
        if 'VGG' in model_type:
            FE = FEVGG
        elif 'ResNet' in model_type:
            FE = FEResNet
        self.feature_extractor = FE(model, target_layer_names)
        self.target_layer_names = target_layer_names
        
    def __call__(self, x, labels=None, relu=True):

        # predition and enumerate features
        feature_maps, pred = self.feature_extractor(x)

        # labels for which NN activates
        if labels is None:
            #cam for pred labels
            _, labels = pred.data.max(dim = 1)

        y_one_hot = to_one_hot(labels, pred.size()[-1]).cuda()
        target_scores = (pred * Variable(y_one_hot)).sum(dim = 1) #(batch, 1)

        self.feature_extractor.model.zero_grad() # reset grad
#        self.feature_extractor.model.features.zero_grad()
#        self.feature_extractor.model.classifier.zero_grad()
        target_scores.backward(retain_variables=True) #store grad

        # sum cam of all layers
        result = np.zeros((x.size()[0], x.size()[2], x.size()[3])) # raw cam
        for feature_map, grad in zip(feature_maps, reversed(self.feature_extractor.gradients)):
            channel_weights = grad.cpu().data.numpy().mean(axis=(2,3), keepdims=True) #b,c,w,h -> b,c,1,1
            wsum = (feature_map.cpu().data.numpy() * channel_weights).sum(axis=1) #b,c,w,h -> b,w,h
            if result is None:
                result = wsum
            else:
                if np.prod(result.shape) > np.prod(wsum.shape):
                    from_arr = wsum
                    to_arr = result
                else:
                    from_arr = result
                    to_arr = wsum
                for i in range(result.shape[0]):
                    result[i,:,:] = to_arr[i,:,:] + cv2.resize(from_arr[i,:,:], to_arr.shape[1:])

        if relu:
            result = np.maximum(result, 0)
            
        # result : b,w,h
            
        # create heatmap-img-arr
        heatmaps = []
        for cam in result:
            cam = cv2.resize(cam, x.size()[2:])
            mask = cam - cam.min()
            mask = cam / cam.max() #0-1
            mask = np.uint8(mask*255)
            heatmap_arr = cv2.applyColorMap(mask, cv2.COLORMAP_JET) #w,h,c 0-255
            heatmaps.append(heatmap_arr)
        
            
        return result, heatmaps
                
        

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image

    path = "/home/ubuntu/prj/auto_dl/data/val/Abyssinian/Abyssinian_122.jpg"
    img = Image.open(path).convert('RGB').resize((224, 224))

    #model = models.resnet18(pretrained=True).cuda()
    #model = models.vgg19(pretrained=True).cuda()
    model = models.resnet18(pretrained=False, num_classes=37)
    #model = models.__dict__["resnet18"](num_classes = 37).cuda()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load("/home/ubuntu/prj/auto_dl/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        #        transforms.Resize(224),
        #        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #   transforms.Lambda(lambda x:x/255.),
        normalize])

    gradcam = GradCam(model, ["layer2","layer3", "layer4"])
    #gradcam = GradCam(model, ["35"])

    input_img = torch.autograd.Variable(preprocess(img).unsqueeze(0)).cuda()
    cam, heatmap = gradcam(input_img, relu=True)

    heatmap_img = Image.fromarray(heatmap[0]).resize((224,224))
    heatmap_img.save("heatmap.png")
#    plt.imshow(np.array(heatmap_img))
    


    
