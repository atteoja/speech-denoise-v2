import os
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalDistortionRatio as SDR
from tqdm import tqdm
from model import SmallCleanUNet
from dataset_class import SpeechTestDataset
from evaluation import get_psnr

from traditional_methods import spectral_subtraction, apply_noisereduce


def test(device, model, test_loader, criterion, output_dir):

    if criterion == "l1":
        criterion = nn.L1Loss()
    elif criterion == "l2":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    trad_spec_losses = []
    trad_spec_psnrs = []
    trad_spec_sdr = []

    trad_nr_losses = []
    trad_psnrs = []
    trad_sdr = []

    test_losses = []
    test_psnrs = []
    test_sdr = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs.unsqueeze(1))

            preds = preds.squeeze(1)
            trad_spec_res = spectral_subtraction(inputs.squeeze(0).cpu().numpy())
            trad_nr_res = apply_noisereduce(inputs.squeeze(0).cpu().numpy())

            trad_spec_res = torch.tensor(trad_spec_res).to(device).unsqueeze(0)
            trad_nr_res = torch.tensor(trad_nr_res).to(device).unsqueeze(0)

            loss = criterion(preds, labels)
            loss_spec = criterion(trad_spec_res.to(device), labels)
            loss_nr = criterion(trad_nr_res.to(device), labels)

            test_losses.append(loss.item())
            trad_spec_losses.append(loss_spec.item())
            trad_nr_losses.append(loss_nr.item())

            test_psnrs.append(get_psnr(preds.cpu(), labels.cpu()))
            trad_spec_psnrs.append(get_psnr(trad_spec_res.cpu(), labels.cpu()))
            trad_psnrs.append(get_psnr(trad_nr_res.cpu(), labels.cpu()))

            sdr_metric = SDR()
            test_sdr.append(sdr_metric(preds.cpu(), labels.cpu()))
            trad_spec_sdr.append(sdr_metric(trad_spec_res.cpu(), labels.cpu()))
            trad_sdr.append(sdr_metric(trad_nr_res.cpu(), labels.cpu()))


            predictions.append(preds.cpu().squeeze(0).squeeze(0).numpy())

            # write the denoised audios
            if i < 5:
                sf.write(f"{output_dir}/pred_{i}.wav", preds.cpu().squeeze(0).squeeze(0).numpy(), 16000)
                sf.write(f"{output_dir}/trad_spec_{i}.wav", trad_spec_res.cpu().squeeze(0).squeeze(0).numpy(), 16000)
                sf.write(f"{output_dir}/trad_nr_{i}.wav", trad_nr_res.cpu().squeeze(0).squeeze(0).numpy(), 16000)
                sf.write(f"{output_dir}/clean_{i}.wav", labels.cpu().squeeze(0).squeeze(0).numpy(), 16000)
                sf.write(f"{output_dir}/noisy_{i}.wav", inputs.cpu().squeeze(0).squeeze(0).numpy(), 16000)

    test_loss = np.array(test_losses).mean()
    test_psnr = np.array(test_psnrs).mean()
    test_sdr = np.array(test_sdr).mean()

    trad_spec_loss = np.array(trad_spec_losses).mean()
    trad_spec_psnr = np.array(trad_spec_psnrs).mean()
    trad_spec_sdr = np.array(trad_spec_sdr).mean()

    trad_nr_loss = np.array(trad_nr_losses).mean()
    trad_nr_psnr = np.array(trad_psnrs).mean()
    trad_nr_sdr = np.array(trad_sdr).mean()

    print(f"Test loss: {test_loss:.2f}, Test PSNR: {test_psnr:.2f}, Test SDR: {test_sdr:.2f}")
    print(f"Traditional spectral subtraction loss: {trad_spec_loss:.2f}, PSNR: {trad_spec_psnr:.2f}, SDR: {trad_spec_sdr:.2f}")
    print(f"Traditional noisereduce loss: {trad_nr_loss:.2f}, PSNR: {trad_nr_psnr:.2f}, SDR: {trad_nr_sdr:.2f}")

    # return set of predictions
    return predictions


def main():
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = 'l1'

    # Make sure to load the model with the same parameters as the training
    unet = SmallCleanUNet(in_channels=1,
                            out_channels=1,
                            depth=2,
                            kernel_size=3)
    unet.to(device)


    unet.load_state_dict(torch.load("saved_models/last_model_22epoch.pth")) # model path

    # Load the test dataset
    test_dataset = SpeechTestDataset(root_dir='.')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define the output directory
    output_dir = "test_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test the model
    preds = test(device=device, model=unet, test_loader=test_loader, criterion=criterion, output_dir=output_dir)

    print(f"Preds len: {len(preds)}")


if __name__ == "__main__":
    main()




