import torch
import torchvision
from pytorch_lightning import LightningModule


class SlotOccupancyClassifier(LightningModule):
    def __init__(self, pretrained, learning_rate):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.learning_rate = learning_rate

        self.model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        n_features_in = self.model.classifier[3].weight.shape[1]
        self.model.classifier[3] = torch.nn.Linear(n_features_in, 2)

    def forward(self, x):
        logits = self.model(x)
        preds = logits.argmax(dim=1)
        return logits, preds

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.model(x)
        batch_losses = self.criterion(logits, y)
        loss = batch_losses.mean()
        return {'loss': loss, 'batch_losses': batch_losses}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([output['batch_losses'] for output in outputs]).mean()
        self.log('valid_loss', avg_loss.item(), prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits, preds = self.forward(x)
        batch_losses = self.criterion(logits, y)
        batch_accuracy = (preds == y)

        return {'batch_losses': batch_losses, 'batch_accuracy': batch_accuracy,
                'predictions': preds, 'label': y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['batch_losses'] for output in outputs]).mean()
        accuracy = torch.stack([output['batch_accuracy'] for output in outputs]).float().mean()

        self.log('valid_loss', avg_loss.item(), prog_bar=True)
        self.log('valid_accuracy', accuracy.item(), prog_bar=True)

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits, preds = self.forward(x)
        batch_losses = self.criterion(logits, y)
        batch_accuracy = (preds == y)

        return {'batch_losses': batch_losses, 'batch_accuracy': batch_accuracy,
                'predictions': preds, 'label': y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['batch_losses'] for output in outputs]).mean()
        accuracy = torch.stack([output['batch_accuracy'] for output in outputs]).float().mean()

        self.log('test_loss', avg_loss.item())
        self.log('test_accuracy', accuracy.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
        return [optimizer], [scheduler]