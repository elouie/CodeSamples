# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        # conv1: convolution layer with 6 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        # conv2: convolution layer with 12 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
        # conv3: convolution layer with 24 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        # fc: fully connected layer with 128 output features
        self.fc = nn.Linear(24*8*8, 128)
        # cls: fully connected layer with 16 output features (the number of classes)
        self.cls = nn.Linear(128, num_classes)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        # ReLU nonlinearity
        x = F.relu(self.conv1(x))
        # ReLU nonlinearity
        x = F.relu(self.conv2(x))
        # ReLU nonlinearity
        x = F.relu(self.conv3(x))
        x = x.view(-1, 24*8*8)
        # ReLU nonlinearity
        x = F.relu(self.fc(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """

    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()
    total_images = labels.data.numpy().size
    # Need to run for each element in input batch
    outputs = net(inputs)
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    # Need to generate num_correct from outputs
    _, predicted = torch.max(outputs, 1)
    num_correct = torch.sum(predicted == labels.data.reshape(-1))

    loss = criterion(outputs, labels.squeeze())
    loss.backward()
    optimizer.step()
    running_loss = loss.item()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        from scipy.ndimage import shift
        x_rand = random.randint(-self.max_shift, self.max_shift)
        y_rand = random.randint(-self.max_shift, self.max_shift)
        image = shift(image, [0, x_rand, y_rand], mode='constant', cval=0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN

        # We need to modify each channel separately
        # First, we need to get a contrast level 'c' from the parameters
        c = random.uniform(self.min_contrast, self.max_contrast)
        # Get the means per channel
        for i in range(image.shape[0]):
            m = np.mean(image[i, :, :])
            image[i, :, :] = ((image[i, :, :] - m) * c) + m
        # We need to shift the image around the means and multiply by the factor
            image = np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Brightness(object):
    """
    Randomly adjusts the brightness of an image. Uniformly select a brightness factor from
    [min_brightness, max_brightness]. Setting the brightness to 0 should set the image to black
    while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_brightness=0.3, max_brightness=1.0):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random brightness
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN

        # First, we need to get a brightness level 'b' from the parameters
        b = random.uniform(self.min_brightness, self.max_brightness)
        image = np.clip(image*b, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        from scipy.ndimage.interpolation import rotate
        angle = random.uniform(-self.max_angle, self.max_angle)
        # Need to avoid interpolating zeroed padding pixels, so order=0.
        image = rotate(image, angle, axes=[1, 2], order=0, reshape=False, mode='constant', cval=0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        if random.random() > self.p:
            image[0,:,:] = np.fliplr(image[0,:,:])
            image[1,:,:] = np.fliplr(image[1,:,:])
            image[2,:,:] = np.fliplr(image[2,:,:])
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

# Noise seems effective on new models, but didn't improve highest performing models.
class Noise(object):
    """
    Randomly adds tiny amounts of noise.

    Inputs:
        e          float in range [0,1]; e variance in noise
    """
    def __init__(self, e=0.5):
        self.e = e

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            noisy_image   image as torch Tensor with added Gaussian noise with variance e.
        """
        image = image.numpy()
        _, H, W = image.shape

        image[0, :, :] = image[0, :, :] + np.random.normal(0, self.e, (H, W))
        image[1, :, :] = image[1, :, :] + np.random.normal(0, self.e, (H, W))
        image[2, :, :] = image[2, :, :] + np.random.normal(0, self.e, (H, W))

        return torch.clamp(torch.Tensor(image), 0, 1)

    def __repr__(self):
        return self.__class__.__name__

# RandomApply was created after we achieved
# 50% accuracy, and trained effectively, but no new models performed better
# with it.
class RandomApply(object):
    """
    Randomly applies a transformation from a list.

    Inputs:
        transforms          transforms to randomly select
    """
    def __init__(self, img_transforms):
        self.img_transforms = img_transforms

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            transformed_image   image to randomly transform
        """
        return random.choice(self.img_transforms)(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # TODO-BLOCK-BEGIN
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # Unused experimental transforms.
        RandomApply([
            Brightness(min_brightness=0.7, max_brightness=1.3),
            Contrast(min_contrast=0.3, max_contrast=2),
            Noise(0.05)
        ]),
        RandomApply([
            Shift(max_shift=16),
            Rotate(max_angle=45)
        ]),
        HorizontalFlip(p=0.5),
        transforms.Normalize(dataset_means, dataset_stds),
    ])
    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN
    batch_size = 16
    epochs = 10
    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer


class AnimalStudentNet(nn.Module):
    # LIL MAX
    # Simple two-layered CNN with argmax to linear
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.1)
        # conv1: convolution layer with 6 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1)
        # conv2: convolution layer with 12 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv2 = nn.Conv2d(6, 9, kernel_size=3, stride=2, padding=1)
        # conv2: convolution layer with 12 output channels, kernel size of 3, stride of 2, padding of 1
        self.conv3 = nn.Conv2d(9, 12, kernel_size=3, stride=2, padding=1)
        # fc: fully connected layer with 128 output features
        self.fc = nn.Linear(12*4*4, 96)
        # cls: fully connected layer with 16 output features (the number of classes)
        self.cls = nn.Linear(96, num_classes)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        x = self.dropout(x)
        # ReLU nonlinearity
        x = F.relu(self.conv1(x))
        # ReLU nonlinearity
        x = F.relu(self.conv2(x))
        # ReLU nonlinearity
        x = self.pool(F.relu(self.conv3(x)))
        # x = F.relu(self.conv3(x))
        # ReLU nonlinearity
        x = x.view(-1, 12*4*4)
        # ReLU nonlinearity
        x = F.relu(self.fc(x))
        x = self.cls(x)
        # TODO-BLOCK-END
        return x


#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    # Retrieve the gradient loss.
    loss = criterion(output, label)
    loss.backward()
    gradient_loss = img.grad
    # Use the sign of the gradient to determine amount of noise to add.
    noise = epsilon * gradient_loss.sign()
    # Remember to clamp the pixels to valid ranges.
    perturbed_image = torch.clamp(img + noise, 0, 1)
    # TODO-BLOCK-END

    return perturbed_image, noise

