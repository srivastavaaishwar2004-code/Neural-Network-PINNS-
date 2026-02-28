import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Mormalization functions for labels and curves
# known ranges from data generation
LABEL_MIN = torch.tensor([150.0, 300.0, 1700.0, 1900.0,  5.0])
LABEL_MAX = torch.tensor([250.0, 500.0, 1900.0, 2200.0, 25.0])

CURVE_MIN = torch.tensor([1.0,  130.0])   
CURVE_MAX = torch.tensor([80.0, 450.0])  

def normalize_label(label):
    return (label - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)

def denormalize_label(label_norm):
    return label_norm * (LABEL_MAX - LABEL_MIN) + LABEL_MIN

def normalize_curve(curve):
    return (curve - CURVE_MIN) / (CURVE_MAX - CURVE_MIN)

class RayleighDataset(Dataset):
    
    def __init__(self, csv_path):
        
        df = pd.read_csv(csv_path)
        
        sample_ids = df['sample_id'].unique()
        
        self.curves = []   # dispersion curves  — input to network
        self.labels = []   # earth model params 
        
        for sid in sample_ids:
            s = df[df['sample_id'] == sid]
            
            curve = torch.tensor(
                s[['freq', 'Vr']].values,
                dtype=torch.float32
            )
            curve = normalize_curve(curve)

            label = torch.tensor(
                s[['vs1','vs2','den1','den2','h1']].iloc[0].values,
                dtype=torch.float32
            )
            label = normalize_label(label)
            self.curves.append(curve)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.curves)
    
    def __getitem__(self, idx):
        return self.curves[idx], self.labels[idx]

if __name__ == '__main__':
    dataset = RayleighDataset('inversion_model/data/pinns_dataset.csv')
    
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
    
    for curves, labels in train_loader:
        print("Curves shape:", curves.shape)
        print("Labels shape:", labels.shape)
        break