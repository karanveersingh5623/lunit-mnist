import mlflow
from torchmetrics import Accuracy
import torch

class Trainer:
    def __init__(self, model, device, lr):
        self.model = model
        self.device = device
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def train(self, dataloader, epoch):
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            accuracy = self.metric_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch
                step = batch // 100 * (epoch + 1)
                mlflow.log_metric("loss", loss, step=step)
                mlflow.log_metric("accuracy", accuracy.item(), step=step)
                print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

    def evaluate(self, dataloader, epoch):
        num_batches = len(dataloader)
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                eval_loss += self.loss_fn(pred, y).item()
                eval_accuracy += self.metric_fn(pred, y)

        eval_loss /= num_batches
        eval_accuracy /= num_batches
        mlflow.log_metric("eval_loss", eval_loss, step=epoch)
        mlflow.log_metric("eval_accuracy", eval_accuracy, step=epoch)

        print(f"Eval metrics: \nAccuracy: {eval_accuracy:.4f}, Avg loss: {eval_loss:.4f} \n")
        return eval_accuracy, eval_loss
