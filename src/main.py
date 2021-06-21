import argparse
import json

import torch

from dataset import CNRExtDataset
from model import ParkingSpotClassifier
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

    dataset = CNRExtDataset(targets_dir=config['dataset']['root_path'],
                            images_dir=config['dataset']['root_path']+config['dataset']['images_dir'],
                            weather=config['dataset']['params']['weather'],
                            camera_ids=config['dataset']['params']['camera_ids'],
                            label_image_size=config['dataset']['params']['label_image_size'],
                            image_size=config['dataset']['params']['image_size'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                 shuffle=False)

    model = torch.hub.load(repo_or_dir=config['model']['repository'], model=config['model']['name'],
                           pretrained=config['model']['pretrained'])
    model.classes = config['model']['classes']

    model.eval()
    for images, targets in dataloader:

        outputs = model(images)
        print(outputs)

if __name__ == '__main__':
    _args = arg_parser.parse_args()
    config_path = _args.conf
    main(config_path)
