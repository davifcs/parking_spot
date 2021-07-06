import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix, classification_report


class SlotOccupancyClassifier(LightningModule):
    def __init__(self, model_repo, model_name, pretrained, learning_rate):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.learning_rate = learning_rate
        self.model = torch.hub.load(repo_or_dir=model_repo, model=model_name, pretrained=pretrained)
        self.model.classifier.out_features = 2

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
        self.log('val_loss', avg_loss.item(), prog_bar=True)

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

        self.log('val_loss', avg_loss.item(), prog_bar=True)
        self.log('val_accuracy', accuracy.item(), prog_bar=True)

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits, preds = self.forward(x)
        batch_losses = self.criterion(logits, y)
        batch_accuracy = (preds == y)

        return {'batch_losses': batch_losses, 'batch_accuracy': batch_accuracy,
                'predictions': preds, 'labels': y}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['batch_losses'] for output in outputs]).mean()
        accuracy = torch.stack([output['batch_accuracy'] for output in outputs]).float().mean()
        preds = torch.stack([output['predictions'] for output in outputs]).float()
        labels = torch.stack([output['labels'] for output in outputs]).float()

        report = classification_report(labels.cpu(), preds.cpu())
        cm = confusion_matrix(labels.cpu(), preds.cpu())

        print('\n', report)
        print('Confusion Matrix \n', cm)
        self.log('test_loss', avg_loss.item())
        self.log('test_accuracy', accuracy.item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
        return [optimizer], [scheduler]