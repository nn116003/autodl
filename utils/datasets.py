import torchvision.datasets as datasets



class ImageFolderWithPath(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPath, self).__getitem__(index), self.imgs[index][0]
