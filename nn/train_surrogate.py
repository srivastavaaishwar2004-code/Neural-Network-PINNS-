import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from surrogate import SurrogateNet
from dataset import LABEL_MIN, LABEL_MAX, CURVE_MIN, CURVE_MAX

# hyperparameters 
DATA_PATH   = 'inversion_model/data/pinns_dataset(2000).csv'
BATCH_SIZE  = 64
EPOCHS      = 500
LR          = 0.001
PATIENCE    = 50
SAVE_PATH   = 'inversion_model/saved_weights/surrogate.pt'


class SurrogateDataset(Dataset):

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # input — normalize each column
        vs1  = torch.tensor(df['vs1'].values,  dtype=torch.float32)
        vs2  = torch.tensor(df['vs2'].values,  dtype=torch.float32)
        den1 = torch.tensor(df['den1'].values, dtype=torch.float32)
        den2 = torch.tensor(df['den2'].values, dtype=torch.float32)
        h1   = torch.tensor(df['h1'].values,   dtype=torch.float32)
        freq = torch.tensor(df['freq'].values,  dtype=torch.float32)
        Vr   = torch.tensor(df['Vr'].values,    dtype=torch.float32)

        # normalize params using same ranges as main dataset
        vs1_n  = (vs1  - LABEL_MIN[0]) / (LABEL_MAX[0] - LABEL_MIN[0])
        vs2_n  = (vs2  - LABEL_MIN[1]) / (LABEL_MAX[1] - LABEL_MIN[1])
        den1_n = (den1 - LABEL_MIN[2]) / (LABEL_MAX[2] - LABEL_MIN[2])
        den2_n = (den2 - LABEL_MIN[3]) / (LABEL_MAX[3] - LABEL_MIN[3])
        h1_n   = (h1   - LABEL_MIN[4]) / (LABEL_MAX[4] - LABEL_MIN[4])

        # normalize freq and Vr using same ranges as curve
        freq_n = (freq - CURVE_MIN[0]) / (CURVE_MAX[0] - CURVE_MIN[0])
        Vr_n   = (Vr   - CURVE_MIN[1]) / (CURVE_MAX[1] - CURVE_MIN[1])

        # stack inputs into shape (8000, 6)
        self.X = torch.stack(
            [vs1_n, vs2_n, den1_n, den2_n, h1_n, freq_n], dim=1
        )

        # output shape (8000, 1)
        self.y = Vr_n.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():

    # data 
    dataset    = SurrogateDataset(DATA_PATH)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"Surrogate training samples: {train_size}")
    print(f"Surrogate validation samples: {val_size}")
    print("-" * 50)

    # model 
    net       = SurrogateNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss  = float('inf')
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):

        # training
        net.train()
        train_loss = 0.0

        for X, y in train_loader:
            pred  = net(X)
            loss  = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        net.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                pred      = net(X)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(net.state_dict(), SAVE_PATH)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best val loss: {best_val_loss:.6f}")
            break

    print("\nSurrogate training complete.")


def evaluate():

    dataset = SurrogateDataset(DATA_PATH)
    net     = SurrogateNet()
    net.load_state_dict(torch.load(SAVE_PATH))
    net.eval()

    print("\n=== Surrogate Evaluation ===")
    print(f"{'freq':>8} {'Vr_true':>10} {'Vr_pred':>10} {'error%':>8}")
    print("-" * 40)

    with torch.no_grad():
        for i in range(10):
            X, y = dataset[i]
            pred = net(X.unsqueeze(0))

            # denormalize to real units
            Vr_true = y.item()   * (CURVE_MAX[1] - CURVE_MIN[1]) + CURVE_MIN[1]
            Vr_pred = pred.item()* (CURVE_MAX[1] - CURVE_MIN[1]) + CURVE_MIN[1]
            freq    = X[5].item()* (CURVE_MAX[0] - CURVE_MIN[0]) + CURVE_MIN[0]
            error   = abs(Vr_true - Vr_pred) / Vr_true * 100

            print(f"{freq:>8.1f} {Vr_true:>10.2f} "
                  f"{Vr_pred:>10.2f} {error:>8.2f}%")


if __name__ == '__main__':
    train()
    evaluate()