import torch
from torch.utils.data import DataLoader
from source import Model, ModelTrain, CSVDataLoader
from logging import basicConfig, debug, DEBUG

# File to train model and save trained model to output


def train():
    basicConfig(level=DEBUG)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    debug(f"Using {device} device")

    train_data_loader = CSVDataLoader("data/train.csv")
    test_data_loader = CSVDataLoader("data/test.csv")

    train_data_loader = DataLoader(train_data_loader, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_data_loader, batch_size=64, shuffle=True)

    model = Model()

    loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=1e-5, weight_decay=0.001, amsgrad=True
    )
    epochs = 20
    for epoch in range(epochs):
        debug(f"Epoch {epoch + 1}\n-------------")

        ModelTrain.train_model(model, train_data_loader, 64, loss_fn, optimizer)

    ModelTrain.test_model(model, test_data_loader, loss_fn)

    debug("Done !")

    torch.save(model.state_dict(), "model_params")


if __name__ == "__main__":
    train()
