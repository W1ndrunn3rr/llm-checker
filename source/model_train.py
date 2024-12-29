from torch import nn
import torch
from sklearn.metrics import f1_score, recall_score, precision_score


class ModelTrain:
    def train_model(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        batch_size: int = 64,
        loss_fn: nn.Module = nn.BCELoss(),
        optimizer: nn.Module = torch.optim.RMSprop,
    ):
        all_preds = []
        all_labels = []
        size = len(data_loader)
        model.train()

        for batch, (X, y) in enumerate(data_loader):
            pred = torch.sigmoid(model(X))
            loss = loss_fn(pred, y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            binary_preds = (pred >= 0.5).float().squeeze()
            all_preds.extend(binary_preds.tolist())
            all_labels.extend(y.tolist())

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        precision = precision_score(
            all_labels, all_preds, average="binary", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        print(
            f"Training Metrics: \n"
            f"Accuracy: {(100*accuracy):>0.1f}% \n"
            f"Precision: {precision:>0.4f}, Recall: {recall:>0.4f}, F1-score: {f1:>0.4f}\n"
        )

    def test_model(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module = nn.BCELoss(),
    ):
        model.eval()
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        test_loss, correct = 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in data_loader:
                pred = torch.sigmoid(model(X))
                test_loss += loss_fn(pred, y.float().unsqueeze(1)).item()

                binary_preds = (pred >= 0.5).float().squeeze()
                correct += (binary_preds == y).sum().item()
                all_preds.extend(binary_preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        test_loss /= num_batches
        correct /= size

        precision = precision_score(
            all_labels, all_preds, average="binary", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

        print(
            f"Test error: \n Accuracy: {(100*correct):>0.1f}% "
            f"Avg. loss: {test_loss:>8f}\n"
        )
        print(
            f"Precision: {precision:>0.4f}, Recall: {recall:>0.4f}, F1-score: {f1:>0.4f}"
        )
        binary_preds = (pred >= 0.5).float()
