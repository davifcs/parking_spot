import argparse
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer

from dataset import CNRExtDataloader, CNRExtPatchesDataset
from model import SlotOccupancyClassifier
import src.utils as utils

arg_parser = argparse.ArgumentParser(description='Run inference with pre-trained network and evaluate')

arg_parser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


def main(config_path):
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if config['mode'] == 'slot_position':
        config_camera_ids = config['dataset']['params']['camera_ids']
        config_slots = config['dataset']['params']['slots']
        gen_target_wh = config['dataset']['params']['gen_target_wh']

        dataloader = CNRExtDataloader(targets_dir=config['dataset']['targets_dir'],
                                images_dir=config['dataset']['images_dir'],
                                targets_origin=config['targets_origin'],
                                weather=config['dataset']['params']['weather'],
                                camera_ids=config['dataset']['params']['camera_ids'],
                                label_image_size=config['dataset']['params']['label_image_size'],
                                image_size=config['dataset']['params']['image_size'],
                                batch_size=config['batch_size'], shuffle=False)

        model = torch.hub.load(repo_or_dir=config['model']['repository'], model=config['model']['name'],
                               pretrained=config['model']['pretrained'])
        model.classes = config['model']['classes']

        n_detections_camera = np.empty((0, 3))
        batch_results = []
        batch_camera_ids = []
        model.eval()
        for images, targets in dataloader:
            images_paths = images[:]
            results = model(images)
            camera_ids = np.array([i[:, 4][0] for i in targets])
            if config['task'] == 'evaluate':
                n_detections, n_targets = utils.detections(results.xyxy, targets)
                n_detections_camera = np.vstack((n_detections_camera, np.hstack((n_detections.reshape(-1, 1),
                                                                                 n_targets.reshape(-1, 1),
                                                                                 camera_ids.reshape(-1, 1)))))
                utils.display_targets_and_detections(images_paths, results.xyxy, targets, n_detections)
            elif config['task'] == 'generate':
                batch_results.extend(results.xywh)
                batch_camera_ids.extend(camera_ids)

        if config['task'] == 'evaluate':
            detection_rate = {}
            for camera in np.unique(n_detections_camera[:, 2]):
                detection_rate[camera] = {}
                detection_rate[camera]['n_detections'] = max(n_detections_camera[n_detections_camera[:, 2] == camera][:, 0])
                detection_rate[camera]['n_targets'] = n_detections_camera[n_detections_camera[:, 2] == camera][0][1]
                detection_rate[camera]['ratio'] = detection_rate[camera]['n_detections']/detection_rate[camera]['n_targets']

            with open("./results/evaluate/detection_rate.txt", "a") as file:
                file.write(str(detection_rate) + "\n")
            print(detection_rate)

        elif config['task'] == 'generate':
            for conf_c, conf_s in zip(config_camera_ids, config_slots):
                detected_xy = []
                for r, c in zip(batch_results, batch_camera_ids):
                    if conf_c == c:
                        detected_xy.extend(r.cpu().numpy()[:, :2])

                clusters_xy = utils.k_means_clusters(np.array(detected_xy), conf_s)
                clusters_xyxy_camera = np.array([clusters_xy[:, 0] - gen_target_wh[0],
                                                 clusters_xy[:, 1] - gen_target_wh[1],
                                                 clusters_xy[:, 0] + gen_target_wh[0],
                                                 clusters_xy[:, 1] + gen_target_wh[1],
                                                 [conf_c] * len(clusters_xy)]).T
                pd.DataFrame(clusters_xyxy_camera).to_csv("./results/generate/camera" + str(conf_c) + ".csv",
                                                          index=False)
    elif config['mode'] == 'slot_occupancy':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['model']['normalize']['mean'], std=config['model']['normalize']['std'])
        ])

        dataset_train = CNRExtPatchesDataset(path=config['dataset']['path'],
                                             cameras_ids=config['train']['camera_ids'],
                                             transform=transform)
        dataset_test = CNRExtPatchesDataset(path=config['dataset']['path'],
                                            cameras_ids=config['test']['camera_ids'],
                                            transform=transform)

        train_size = int(len(dataset_train) * (1 - config['train']['val_size']))
        val_size = len(dataset_train) - train_size
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [train_size, val_size])

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=config['train']['batch_size'], shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=config['train']['batch_size'], shuffle=False)
        dataloader_test = DataLoader(dataset=dataset_test, shuffle=False)

        pl_model = SlotOccupancyClassifier(pretrained=config['model']['pretrained'],
                                           learning_rate=config['model']['learning_rate'])
        trainer = Trainer(gpus=config['gpus'], max_epochs=config['epochs'])
        trainer.fit(model=pl_model, train_dataloader=dataloader_train, val_dataloaders=dataloader_val)
        trainer.test(pl_model, dataloader_test)


if __name__ == '__main__':
    _args = arg_parser.parse_args()
    config_path = _args.conf
    main(config_path)
