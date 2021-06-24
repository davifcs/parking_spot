import argparse
import json

import torch
import numpy as np

from dataset import CNRExtDataloader
import src.utils as utils


arg_parser = argparse.ArgumentParser(description='Run inference with pre-trained network and evaluate')

arg_parser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')


def main(config_path):
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if config['mode'] == "detection":
        dataloader = CNRExtDataloader(targets_dir=config['dataset']['root_path'],
                                images_dir=config['dataset']['root_path']+config['dataset']['images_dir'],
                                weather=config['dataset']['params']['weather'],
                                camera_ids=config['dataset']['params']['camera_ids'],
                                label_image_size=config['dataset']['params']['label_image_size'],
                                image_size=config['dataset']['params']['image_size'],
                                batch_size=config['batch_size'], shuffle=False)

        model = torch.hub.load(repo_or_dir=config['model']['repository'], model=config['model']['name'],
                               pretrained=config['model']['pretrained'])
        model.classes = config['model']['classes']

        n_detections_camera = np.empty((0, 3))
        detections_xy_camera = np.empty((0, 3))

        model.eval()
        for images, targets in dataloader:
            images_paths = images[:]
            results = model(images)
            camera_ids = np.array([i[:, 5][0] for i in targets])
            if config['task'] == "evaluate":
                n_detections, n_targets = utils.detections(results.xyxy, targets)
                n_detections_camera = np.vstack((n_detections_camera, np.hstack((n_detections.reshape(-1, 1),
                                                                                 n_targets.reshape(-1, 1),
                                                                                 camera_ids.reshape(-1, 1)))))
                utils.display_targets_and_detections(images_paths, results.xyxy, targets, n_detections)

        if config['task'] == "evaluate":
            detection_rate = {}
            for camera in np.unique(n_detections_camera[:, 2]):
                detection_rate[camera] = {}
                detection_rate[camera]['n_detections'] = max(n_detections_camera[n_detections_camera[:, 2] == camera][:, 0])
                detection_rate[camera]['n_targets'] = n_detections_camera[n_detections_camera[:, 2] == camera][0][1]
                detection_rate[camera]['ratio'] = detection_rate[camera]['n_detections']/detection_rate[camera]['n_targets']

            with open("./results/detection_rate.txt", "a") as file:
                file.write(str(detection_rate) + "\n")
            print(detection_rate)

if __name__ == '__main__':
    _args = arg_parser.parse_args()
    config_path = _args.conf
    main(config_path)
