import torch

class ParkingSpotClassifier():
    def __init__(self, repository, name, pretrained, classes):
        self.model = torch.hub.load(repo_or_dir=repository, model=name, pretrained=pretrained)
        self.model.classes = classes

    def inference(self, images):
        results = self.model(images)
        return results


