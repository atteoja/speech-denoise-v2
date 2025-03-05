import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DataParallel
import time
from datetime import datetime, timedelta
import soundfile as sf
import librosa
from torch.optim.lr_scheduler import MultiStepLR

import pathlib
from model import SmallCleanUNet
from dataset_class import SpeechTestDataset, SpeechTrainDataset
from evaluation import get_psnr
from utils import get_files_from_dir


def train(device, model, train_loader, val_loader,
          epochs=200,
          criterion="l1",
          optimizer="adam",
          lr=1e-3,
          save_path="saved_models",
          loss_increase_min = 1e-2,
          patience = 5
          ):

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    if optimizer == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    if criterion == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()

    scheduler = MultiStepLR(optimizer=optimizer, milestones=[100], gamma=0.1)

    prev_loss = float(10000)
    patience_counter = 0

    checkpoint_path = "model_checkpoint"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    for epoch in tqdm(range(epochs)):
        #epoch_start_time = time.time()
        train_loss_epoch = []
        val_loss_epoch = []

        model.train()

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            train_loss_epoch.append(loss.item())

        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)
                preds = model(inputs)


                loss = criterion(preds, labels)

                val_loss_epoch.append(loss.item())

        train_loss = np.array(train_loss_epoch).mean()
        val_loss = np.array(val_loss_epoch).mean()

        scheduler.step()

        print('\n\n', f" *** Epoch {epoch:03d} ***\n Train loss: {train_loss:.3f}\n Validation loss: {val_loss:.3f}\n Learning rate: {optimizer.param_groups[0]['lr']}\n") #\n Time: {format_time(time.time() - epoch_start_time)}

        if epoch != 0 and epoch % 2 == 0:

            prev_files = get_files_from_dir(checkpoint_path)
            if len(prev_files) > 0:
                for file in prev_files:
                    pathlib.Path.unlink(file)

            model_checkpoint_name = checkpoint_path + f"/model_ckpt_{epoch+1}.pth"
            torch.save(model.state_dict(), model_checkpoint_name)

        if train_loss > prev_loss or abs(train_loss - prev_loss) < loss_increase_min:
            patience_counter += 1
        else:
            patience_counter = 0
        
        prev_loss = train_loss
        
        if patience_counter == patience:
            print("Early stopping at epoch ", epoch + 1, ".")
            break

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, "last_model.pth")
    torch.save(model.state_dict(), model_path) # save trained model

    return model

def test(device, model, test_loader, criterion):

    test_losses = []
    test_psnrs = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)
            preds = model(inputs)

            loss = criterion(preds, labels)

            test_losses.append(loss.item())
            test_psnrs.append(get_psnr(preds.cpu(), labels.cpu()))
            predictions.append(preds.cpu())
        
    test_loss = np.array(test_losses).mean()
    test_psnr = np.array(test_psnrs).mean()
    print(f"Test loss: {test_loss}")
    print(f"Test PSNR: {test_psnr}")

    # return set of predictions
    return predictions

def collate_fn(batch):
    # Find max length in the batch
    max_len = max([x[0].shape[1] for x in batch])
    
    # Prepare lists for inputs and labels
    inputs = []
    labels = []
    
    # Pad each sequence to max_len
    for input_seq, label_seq in batch:
        curr_len = input_seq.shape[1]
        
        # Calculate padding
        pad_len = max_len - curr_len
        
        # Pad the sequences
        if pad_len > 0:
            pad_width = ((0, 0), (0, pad_len))
            input_padded = np.pad(input_seq, pad_width, mode='constant', constant_values=0)
            label_padded = np.pad(label_seq, pad_width, mode='constant', constant_values=0)
        else:
            input_padded = input_seq
            label_padded = label_seq
            
        inputs.append(input_padded)
        labels.append(label_padded)
    
    # Convert to torch tensors
    inputs = torch.FloatTensor(np.stack(inputs))
    labels = torch.FloatTensor(np.stack(labels))
    
    return inputs, labels

def main():

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using {n_gpu} GPU(s)")

    train_dataset = SpeechTrainDataset(root_dir='.')

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    batch_size = 32
    criterion = 'l1'
    optimizer = 'adam'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = SpeechTestDataset(root_dir='.')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("\nData loaded. Train size: ", len(train_dataset), " Val size: ", len(val_dataset), " Test size: ", len(test_dataset), "\n")

    unet = SmallCleanUNet(in_channels=1,
                          out_channels=1,
                          depth=2,
                          kernel_size=5)
    unet.to(device)
    
    unet = train(device=device, model=unet,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  epochs=200,
                  criterion=criterion,
                  optimizer=optimizer,
                  lr=1e-3)
    
    
    test(device=device, model=unet, test_loader=test_loader, criterion=criterion)


if __name__ == "__main__":
    main()

    """# load the trained model
    device = torch.device("cuda")

    unet = SmallCleanUNet(in_channels=1,
                          out_channels=1,
                          depth=4,
                          kernel_size=5)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        unet = DataParallel(unet)

    unet.load_state_dict(torch.load("saved_models/last_model.pth"))

    # Load the test dataset
    test_dataset = SpeechTestDataset(root_dir='.', features=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Test the unet
    for i, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.unsqueeze(1), labels.unsqueeze(1)
        preds = unet(inputs)
        print(i)
        if i == 10:
            break

    
    print("Model testing finished.")"""