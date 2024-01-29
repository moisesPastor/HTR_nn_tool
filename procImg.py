# Standard packages
import torchvision.transforms as transforms
import torchvision.transforms.functional as Fv
from PIL import ImageOps

class FixedHeightResize(object):
    """
    from https://github.com/pytorch/vision/issues/908
    """
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        size = (self.height, self._calc_new_width(img))
        return Fv.resize(img, size)

    def _calc_new_width(self, img):
        old_width, old_height = img.size
        aspect_ratio = old_width / old_height
        return round(self.height * aspect_ratio)


# Transformations applied for image processing
# https://pytorch.org/vision/main/transforms.html
def get_tranform(fixedHeight=64, dataAug=False):

    # List of transformations for Data Augmentation
    daTr = [
             # Random AdjustSharpness
             #transforms.RandomAdjustSharpness(1, p=0.5),
             # Random Rotation
             #transforms.RandomRotation(1,fill=255),
             # Random Affine Transformations (this includes Rotation)
             transforms.RandomAffine(degrees=1,
                                     translate=(0.005,0.05),
                                     shear=1,
                                     fill=255),
             # Random Gaussian Blur
             #transforms.GaussianBlur(3, sigma=(0.001, 1.0))
           ] if dataAug else []

    # List of mandatory transformations
    nTr = [
        # Invert pixel values
        transforms.Lambda(lambda x: ImageOps.invert(x)),
        # Scale to a fixed height
        FixedHeightResize(fixedHeight),
        # Convert a (PIL) image to tensor
        transforms.ToTensor()
        ]

    return transforms.Compose( daTr + nTr )
