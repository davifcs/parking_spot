import argparse
import json

from dataset import DatasetAssemble
from model import ParkingSpotClassifier
from src.utils import compute_iou


arg_parser = argparse.ArgumentParser(description='Run inference with pre-trained network and evaluate')

arg_parser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')


def main(config_path):
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    dataset = DatasetAssemble(labels_path=config['dataset']['root_path'],
                              images_path=config['dataset']['root_path']+config['dataset']['images_dir'],
                              weather=config['dataset']['params']['weather'],
                              camera_ids=config['dataset']['params']['camera_ids'])

    model_ = ParkingSpotClassifier(repository=config['model']['repository'], name=config['model']['name'],
                                   classes=config['model']['classes'], pretrained=config['model']['pretrained'])

    results = model_.inference(dataset.images_list[:10])
    dataset_xywh = dataset.labels_df[['X', 'Y', 'W', 'H']].values
    iou = compute_iou(results.xywh, dataset_xywh)
    print(iou)



if __name__ == '__main__':
    _args = arg_parser.parse_args()
    config_path = _args.conf
    main(config_path)
