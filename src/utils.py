import torch
import torchvision.ops.boxes as bops


def compute_iou(predicted, label):
    box1 = predicted
    box2 = torch.tensor(label, dtype=torch.float)
    return bops.box_iou(box1, box2)
