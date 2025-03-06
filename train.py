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
from modules import get_psnr, L1STFTLoss
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
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    if criterion == "l1":
        criterion = nn.L1Loss()
    elif criterion == "l2":
        criterion = nn.MSELoss()
    elif criterion == "l1_stft":
        criterion = L1STFTLoss(n_fft=2048,
                                hop_length=512,
                                use_log=True)
    else:
        criterion = nn.L1Loss()

    scheduler = MultiStepLR(optimizer=optimizer, milestones=[25], gamma=0.1)

    prev_loss = float(10000)
    patience_counter = 0

    checkpoint_path = "model_checkpoint"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    for epoch in tqdm(range(1, epochs+1)):
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

        if epoch != 1 and epoch % 2 == 0:

            prev_files = get_files_from_dir(checkpoint_path)
            if len(prev_files) > 0:
                for file in prev_files:
                    pathlib.Path.unlink(file)

            model_checkpoint_name = checkpoint_path + f"/model_ckpt_{epoch}.pth"
            torch.save(model.state_dict(), model_checkpoint_name)

        if train_loss > prev_loss or abs(train_loss - prev_loss) < loss_increase_min:
            patience_counter += 1
        else:
            patience_counter = 0
        
        prev_loss = train_loss
        
        if patience_counter == patience:
            print("Early stopping at epoch ", epoch, ".")
            break

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, "last_model.pth")
    torch.save(model.state_dict(), model_path) # save trained model

    return model

def test(device, model, test_loader, criterion):

    if criterion == "l1":
        criterion = nn.L1Loss()
    elif criterion == "l2":
        criterion = nn.MSELoss()
    elif criterion == "l1_stft":
        criterion = L1STFTLoss(n_fft=2048,
                                hop_length=512,
                                use_log=True)
    else:
        criterion = nn.L1Loss()

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
            predictions.append(preds.cpu().squeeze(0).squeeze(0).numpy())

    test_loss = np.array(test_losses).mean()
    test_psnr = np.array(test_psnrs).mean()
    print(f"Test loss: {test_loss}")
    print(f"Test PSNR: {test_psnr}")

    # return set of predictions
    return predictions

def main():


    # DEFINE HYPERPARAMS

    training = True
    device = None

    batch_size = 16
    epochs = 4
    criterion = 'l1_stft'        # l1, l2, l1_stft (L1 + L1 STFT loss)
    optimizer = 'adam'      # adam, adamw
    lr = 1e-3

    # HYPERPARAMS END


    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using {n_gpu} GPU(s)")
    
    if device is None:
        device = torch.device("cpu")
        print("Using CPU.")

    unet = SmallCleanUNet(in_channels=1,
                          out_channels=1,
                          depth=2,
                          kernel_size=3)
    unet.to(device)

    # Set training = True to train the model
    if training:
        train_dataset = SpeechTestDataset(root_dir='.')

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        print("\nData loaded. Train size: ", len(train_dataset), " Val size: ", len(val_dataset), "\n")

        unet = train(device=device,
                    model=unet,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr=lr)

    test_dataset = SpeechTestDataset(root_dir='.')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    preds = test(device=device, model=unet, test_loader=test_loader, criterion=criterion)

    test_wavs_path = "test_sounds"
    if not os.path.exists(test_wavs_path):
        os.mkdir(test_wavs_path)
    
    for file in get_files_from_dir(test_wavs_path):
        pathlib.Path.unlink(file)

    for i in range(3):
        print(f"\nSaving test_{i+1}.wav\nMax: {preds[i].max()}\nMin: {preds[i].min()}")
        sf.write(f"{test_wavs_path}/test_{i+1}.wav", preds[i], samplerate=22050)


if __name__ == "__main__":
    main()