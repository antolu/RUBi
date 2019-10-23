
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

def compute_acc(labels, predictions):
    """
    predictions: output of a forward pass through the model
    computes the accuracy by comparing them with the correct labels
    """
    softmax = nn.Softmax(dim=1)

    predictions = softmax(predictions["logits"])

    correct = (torch.max(predictions, dim=1) == labels).float().sum()

    return correct


def get_attention_mask(input_data, predictions):
    # TODO: Need to add boxes to dataloader

    crop = input_data["boxes"][predictions["argmax"]]

    orig_img = input_data["img_embed"].permute(1, 2, 0).numpy()

    s_img = # DO SOME CROPPING
    l_img = cv2.imread("larger_image.jpg")
    x_offset = y_offset = 50
    orig_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    raise NotImplementedError()


def disp_tensor(tnsr):
    """
    Display a PyTorch tensor as an image
    """
    img = tnsr.permute(1, 2, 0).numpy()

    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.clf()

