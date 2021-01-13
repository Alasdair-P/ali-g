import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
import numbers
import random
import warnings

from PIL import Image
import torch
import torchvision.transforms.functional as F
from torch import Tensor

from typing import Tuple, List, Optional


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomHorizontalFlipIndex(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        # def __call__(self, sample):
        # print('sample', sample)
        trans, img = sample['trans'], sample['image']
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return {'trans':torch.tensor([0]), 'image': F.hflip(img)}
        return {'trans':torch.tensor([1]), 'image': img}

    def __repr__(self):
       return self.__class__.__name__ + '(p={})'.format(self.p)

class ToTensorIndex(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        print('sample', sample)
        trans, img = sample['trans'], sample['image']
        return {'trans':trans, 'image': F.to_tensor(img)}

class RandomCropIndex(torch.nn.Module):
    """Crop the given image at a random location.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            Mode symmetric is not yet supported for Tensor inputs.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
         """
        w, h = img.size
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, sample):
        # print(sample)
        trans, img = sample['trans'], sample['image']
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
         """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # width, height = F._get_image_size(img)
        # width, height, _ = img.shape
        width, height = img.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return {'trans':torch.cat((trans, torch.tensor([i, j])),dim=0), 'image': F.crop(img, i, j, h, w)}

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

class HorizontalFlipIndex(torch.nn.Module):
    def __init__(self, flip=0):
        super().__init__()
        self.flip = flip

    def forward(self, sample):
        trans, img = sample['trans'], sample['image']
        if self.flip:
            return {'trans':torch.tensor([1]), 'image': F.hflip(img)}
        return {'trans':torch.tensor([0]), 'image': img}

    def __repr__(self):
       return self.__class__.__name__ + '(p={})'.format(self.p)


class CropIndex(torch.nn.Module):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", crop_index_i=0, crop_index_j=0):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.i = crop_index_i
        self.j = crop_index_j

    def forward(self, sample):
        # print(sample)
        trans, img = sample['trans'], sample['image']
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
         """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = img.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        h, w = self.size
        i, j = self.i, self.j

        return {'trans':torch.cat((trans, torch.tensor([i, j])),dim=0), 'image': F.crop(img, i, j, h, w)}

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class NormalizeCifar(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print('sample', sample)
        trans, img = sample['trans'], sample['image']
        means = [125.3, 123.0, 113.9]
        stds = [63.0, 62.1, 66.7]
        normalize = transforms.Normalize([x / 255.0 for x in means],
                                         [x / 255.0 for x in stds])
        return {'trans':trans, 'image': normalize(img)}

class ToTensorIndex(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trans, img = sample['trans'], sample['image']
        return {'trans':trans, 'image': F.to_tensor(img)}

class CreateTransDict(object):
    def __call__(self, img):
        return {'trans':[], 'image': img}

class FormatTransDict(object):
    def __call__(self, sample):
        # print('sample', sample)
        trans, img = sample['trans'], sample['image']
        return {'trans':torch.cat(trans), 'image': img}

if __name__ == "__main__":
    test_dict = {'trans': [torch.arange(2),torch.arange(3)],'image': 2}
    a = FormatTransDict()
    print(a(test_dict))
