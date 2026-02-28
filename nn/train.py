import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import RayleighDataset, denormalize_label
from model import RayleighNet
from loss import load_surrogate, total_loss

# hyperparameters 
DATA_PATH    = 'inversion_model/data/pinns_dataset(2000).csv'
BATCH_SIZE   = 16
EPOCHS       = 500
LR           = 0.001
LAMBDA       = 0.001
PATIENCE     = 50     
SAVE_PATH    = 'inversion_model/saved_weights/best_model_2000.pt'

def train():
    
    dataset    = RayleighDataset(DATA_PATH)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    # model and optimizer 
    net       = RayleighNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    train_losses   = []
    val_losses     = []
    best_val_loss  = float('inf')
    patience_count = 0
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}, Lambda: {LAMBDA}")
    print("-" * 60)
    
    surrogate = load_surrogate('inversion_model/saved_weights/surrogate.pt')
    for epoch in range(1, EPOCHS + 1):
        
        # training 
        net.train()  
        epoch_train_loss = 0.0
    
        for curves, labels in train_loader:
            
            # forward pass
            predicted = net(curves)
            
            # compute loss
            loss, d_loss, p_loss = total_loss(
            predicted, labels, curves, surrogate, LAMBDA
            )
            
            # backward pass
            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()        
            
            epoch_train_loss += loss.item()
        
        # average over batches
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # validation
        net.eval()  
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for curves, labels in val_loader:
                predicted        = net(curves)
                loss, d, p = total_loss(
                    predicted, labels, curves, surrogate, LAMBDA
                )
                epoch_val_loss  += loss.item()
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # logging 
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | "
                f"Train {epoch_train_loss:.4f} | "
                f"d_loss {d_loss:.4f} | "
                f"p_loss {p_loss:.4f}")
        
        # save best model 
        if epoch_val_loss < best_val_loss:
            best_val_loss  = epoch_val_loss
            patience_count = 0
            torch.save(net.state_dict(), SAVE_PATH)
        else:
            patience_count += 1
        
        # early stopping 
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # plot training curves 
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses,   label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/training_curves.png', dpi=150)
    plt.close()
    print("\nTraining curves saved.")

def evaluate():

    dataset = RayleighDataset(DATA_PATH)
    net     = RayleighNet()
    net.load_state_dict(torch.load(SAVE_PATH))
    net.eval()
    
    print("\n=== Evaluation on 5 samples ===")
    print(f"{'':>10} {'vs1':>8} {'vs2':>8} {'den1':>8} {'den2':>8} {'h1':>8}")
    print("-" * 55)
    
    with torch.no_grad():
        for i in range(5):
            curve, label = dataset[i]
            curve = curve.unsqueeze(0)
            
            predicted_norm = net(curve)
            
            # denormalize both to real units
            predicted_real = denormalize_label(predicted_norm).squeeze()
            true_real      = denormalize_label(label)
            
            print(f"True      {i+1}: "
                  f"{true_real[0]:>8.1f} "
                  f"{true_real[1]:>8.1f} "
                  f"{true_real[2]:>8.1f} "
                  f"{true_real[3]:>8.1f} "
                  f"{true_real[4]:>8.1f}")
            print(f"Predicted {i+1}: "
                  f"{predicted_real[0]:>8.1f} "
                  f"{predicted_real[1]:>8.1f} "
                  f"{predicted_real[2]:>8.1f} "
                  f"{predicted_real[3]:>8.1f} "
                  f"{predicted_real[4]:>8.1f}")
            print()

def evaluate_metrics(save_path, data_path):

    dataset = RayleighDataset(data_path)
    net     = RayleighNet()
    net.load_state_dict(torch.load(save_path))
    net.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for i in range(len(dataset)):
            curve, label = dataset[i]
            curve = curve.unsqueeze(0)

            predicted_norm = net(curve)
            predicted_real = denormalize_label(predicted_norm).squeeze()
            true_real      = denormalize_label(label)

            all_true.append(true_real.numpy())
            all_pred.append(predicted_real.numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    params = ['vs1', 'vs2', 'den1', 'den2', 'h1']
    
    print("\n=== Full Dataset Metrics ===")
    print(f"{'param':>6} {'MAE':>10} {'MAPE%':>10}")
    print("-" * 30)
    
    for i, p in enumerate(params):
        mae  = np.mean(np.abs(all_true[:,i] - all_pred[:,i]))
        mape = np.mean(np.abs(all_true[:,i] - all_pred[:,i]) / 
                       np.abs(all_true[:,i])) * 100
        print(f"{p:>6} {mae:>10.3f} {mape:>10.2f}%")

if __name__ == '__main__':

    dataset = RayleighDataset(DATA_PATH)

    curve1, label1 = dataset[0]
    curve2, label2 = dataset[1]

    print("Curve 1 first 5 Vr values:", curve1[:5, 1])
    print("Curve 2 first 5 Vr values:", curve2[:5, 1])
    print("Label 1:", denormalize_label(label1))
    print("Label 2:", denormalize_label(label2))
    train()
    evaluate()
    evaluate_metrics(SAVE_PATH, DATA_PATH)
