import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


def k_means_clusters(xy, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xy)
    return kmeans.cluster_centers_

def detections(detected, target):
    n_detections = []
    n_targets = []
    for d, t in zip(detected, target):
        t_ = np.array(t)
        t_xyxy = t_[:, :4]
        if len(d) > 0:
            d_xy = []
            center = (d[:, :2] + d[:, 2:4])//2
            d_xy.extend(center.cpu().tolist())
            d_xy = np.array(d_xy)
            n_detections.append(sum(((d_xy >= t_xyxy[:, None, :2]) & (d_xy <= t_xyxy[:, None, 2:])).all(2).any(1)))
            n_targets.append(len(t_xyxy))
        else:
            n_detections.append(0)
            n_targets.append(len(t_xyxy))
    return np.array(n_detections), np.array(n_targets)


def display_targets_and_detections(image_path, detected, target, n_detections):
    for p, d, t, n in zip(image_path, detected, target, n_detections):
        image = Image.open(p).convert('RGB')
        draw = ImageDraw.Draw(image)
        output_path = os.path.abspath('.') + '/results/evaluate/' + \
                      os.path.basename(os.path.split(p)[0]) + '/' + os.path.basename(p)
        for d_bbox in d:
            draw.rectangle(d_bbox[:4].cpu().detach().numpy(), outline='red', width=2)
            draw.line(d_bbox.cpu().detach().numpy()[:4], fill='red', width=2)
            draw.line(np.hstack([d_bbox.cpu().detach().numpy()[:4:3], np.flip(d_bbox.cpu().detach().numpy()[1:3])]),
                      fill='red', width=2)
        for t_bbox in t:
            draw.rectangle(t_bbox[:4], outline='green', width=2)
        font = ImageFont.truetype("arial.ttf", 32)
        draw.text((0, 0), str(n) + '/' + str(len(t)), (255, 0, 0), font=font)
        image.save(output_path)
