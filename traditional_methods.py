import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

from dataset_class import SpeechTestDataset


def spectral_subtraction(noisy_audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Applies the spectral subtraction to the noisy signal."""
    
    # Magnitude spectrum
    noisy_stft = librosa.stft(noisy_audio)
    mag, phase = np.abs(noisy_stft), np.angle(noisy_stft)

    # Estimate that the first 0.5 seconds of the signal contains only noise
    noise_estimate = noisy_audio[:int(0.5 * sr)]
    est_mag = np.abs(librosa.stft(noise_estimate))
    # Apply spectral subtraction
    magnitude_denoised = np.maximum(mag - est_mag.mean(axis=1, keepdims=True), 0)

    # Reconstruct denoised signal using original phase
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    clean_audio = librosa.istft(stft_denoised)

    # Ensure the output length matches the input length
    if len(clean_audio) > len(noisy_audio):
        clean_audio = clean_audio[:len(noisy_audio)]
    else:
        clean_audio = np.pad(clean_audio, (0, len(noisy_audio) - len(clean_audio)), mode='constant')
        
    return clean_audio


def apply_noisereduce(noisy_audio: np.ndarray, sr:int = 16000) -> np.ndarray:
    """Apply the noisereduce library to the noisy audio."""
    return nr.reduce_noise(y=noisy_audio, sr=16000)
    

def apply_denoising(output_dir: str, test_dataset: SpeechTestDataset, n: int = 10, sr: int = 48000):
    """Apply the denoising model to the test dataset.
    Save the denoised audio files to the disk.
    """
    for i in range(n):
        noisy, clean = test_dataset[i]
        spectral = spectral_subtraction(noisy)
        nr_denoised = nr.reduce_noise(y=noisy, sr=sr)
        os.makedirs(output_dir, exist_ok=True)
        sf.write(f"{output_dir}/spectral_{i}.wav", spectral, sr)
        sf.write(f"{output_dir}/nr_{i}.wav", nr_denoised, sr)
        sf.write(f"{output_dir}/original_noisy_{i}.wav", noisy, sr)


def main():
    # Load the test dataset
    test_dataset = SpeechTestDataset(root_dir='.', features=False)
    
    # Apply denoising
    apply_denoising(output_dir="traditional_results", test_dataset=test_dataset, n=10)


if __name__ == "__main__":
    main()

# All data is sampled
# at 48 kHz and orthographic transcription is also available
# source to valentini paper: https://www.isca-archive.org/interspeech_2016/valentinibotinhao16_interspeech.pdf
