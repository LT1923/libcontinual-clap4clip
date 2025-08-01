import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

def random_update(datasets, buffer):

    images = np.array(datasets.images + buffer.images)
    labels = np.array(datasets.labels + buffer.labels)
    perm = np.random.permutation(len(labels))

    images, labels = images[perm[:buffer.buffer_size]], labels[perm[:buffer.buffer_size]]

    buffer.images = images.tolist()
    buffer.labels = labels.tolist()

def herding_update(datasets, buffer, feature_extractor, device):

    print("Using Herding Update Strategy")

    per_classes = buffer.buffer_size // buffer.total_classes

    selected_images, selected_labels = [], []
    images = np.array(datasets.images + buffer.images)
    labels = np.array(datasets.labels + buffer.labels)

    for cls in range(buffer.total_classes):
        if cls % 20 == 0 or cls == buffer.total_classes-1:
            print("Construct examplars for class {}".format(cls))
        cls_images_idx = np.where(labels == cls)
        cls_images, cls_labels = images[cls_images_idx], labels[cls_images_idx]

        cls_selected_images, cls_selected_labels = construct_examplar(copy.copy(datasets), cls_images, cls_labels, feature_extractor, per_classes, device)
        selected_images.extend(cls_selected_images)
        selected_labels.extend(cls_selected_labels)


    buffer.images, buffer.labels = selected_images, selected_labels

def construct_examplar(datasets, images, labels, feature_extractor, per_classes, device):
    if len(images) <= per_classes:
        return images, labels
    
    datasets.images, datasets.labels = images, labels
    dataloader = DataLoader(datasets, shuffle = False, batch_size = 32, drop_last = False)

    with torch.no_grad():
        features = []
        for data in dataloader:
            imgs = data['image'].to(device)
            img_feat = feature_extractor(imgs)
            features.append(img_feat.cpu().numpy().tolist())

    features = np.concatenate(features)
    selected_images, selected_labels = [], []
    selected_features = []
    class_mean = np.mean(features, axis=0)

    for k in range(1, per_classes+1):
        if len(selected_features) == 0:
            S = np.zeros_like(features[0])
        else:
            S = np.mean(np.array(selected_features), axis=0)


        mu_p = (S + features) / k
        i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

        selected_images.append(images[i])
        selected_labels.append(labels[i])
        selected_features.append(features[i])

        features = np.delete(features, i, axis=0) 
        images = np.delete(images, i)
        labels = np.delete(labels, i)

    return selected_images, selected_labels

"""
def balance_random_update(dataset, buffer):
    images = np.array(datasets.images + buffer.images)
    labels = np.array(datasets.labels + buffer.labels)
    perm = np.random.permutation(len(labels))

    exemplar_per_class = buffer.buffer_size // len(labels)  # cifar100的类别数不会超过2000
    exemplars = []
    for label in labels:
        _exemplars =  # 怎么采样？
        exemplars.append(_exemplars)
    
    buffer.images = exemplars.tolist()
    # images, labels = images[perm[:buffer.buffer_size]], labels[perm[:buffer.buffer_size]]

    # buffer.images = images.tolist()
    buffer.labels = labels.tolist()
"""

def balance_random_update(dataset, buffer):
    # todo: how to store images and labels as tensor instead of tuple?
    images = np.array(dataset.images + buffer.images)
    labels = np.array(dataset.labels + buffer.labels)
    
    perm = np.random.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]
    
    unique_labels = np.unique(labels)
    exemplar_per_class = buffer.buffer_size // len(unique_labels)
    
    exemplars = []
    exemplar_labels = []
    
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        if len(class_indices) > exemplar_per_class:
            selected_indices = np.random.choice(class_indices, exemplar_per_class, replace=False)
        else:
            selected_indices = class_indices
        exemplars.extend(images[selected_indices])
        exemplar_labels.extend(labels[selected_indices])
    
    if len(exemplars) > buffer.buffer_size:
        selected_indices = np.random.choice(len(exemplars), buffer.buffer_size, replace=False)
        exemplars = np.array(exemplars)[selected_indices]
        exemplar_labels = np.array(exemplar_labels)[selected_indices]
    
    buffer.images = exemplars
    buffer.labels = exemplar_labels