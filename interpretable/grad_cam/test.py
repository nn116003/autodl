import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model = models.__dict__['resnet18'](num_classes=3)
#model = models.vgg19()
model = torch.nn.DataParallel(model).cuda()
#for c in list(model.children()):
#    print(c)

#for name, module in model.features._modules.items():
#    print(name)
#    print(module)

for name, module in model.modules().items():
    print(name)

