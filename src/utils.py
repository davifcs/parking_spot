import numpy as np
from sklearn.cluster import KMeans
import torch
import torchvision.ops.boxes as bops


def compute_iou(predicted, label):
    box1 = predicted
    box2 = torch.tensor(label, dtype=torch.float)
    return bops.box_iou(box1, box2)


def k_means_clusters(output, n_clusters):
    detected_xy = []
    for o in output:
        detected_xy.extend(o[:, :2].cpu().tolist())
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(detected_xy)
    return kmeans.cluster_centers_


def compute_accuracy(output, label):
    label_boxes = np.array([label[:, 0], label[:, 1], (label[:, 0] + label[:, 2]), (label[:, 1] + label[:, 3])]).T
    detected_xy = []
    for o in output:
        detected_xy.extend(o[:, :2].cpu().tolist())
    detected_xy = np.array(detected_xy)
    return ((detected_xy >= label_boxes[:, None, :2]) & (detected_xy <= label_boxes[:, None, 2:])).all(2).any(1).mean()
