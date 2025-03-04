import os
import librosa
import numpy as np
import soundfile as sf

from dataset_class import SpeechTestDataset


def wiener_filtering(noisy: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Applies the Wiener filter to the noisy signal."""
    
    # Compute STFT for noisy and clean signals
    noisy_stft = librosa.stft(noisy)
    clean_stft = librosa.stft(clean)

    # Power spectrum
    noisy_mag = np.abs(noisy_stft)
    clean_mag = np.abs(clean_stft)

    # apply wiener filter
    wiener_filter = clean_mag ** 2 / (clean_mag ** 2 +  noisy_mag ** 2)
    filtered_stft = wiener_filter * noisy_stft

    # synthesize the signal
    filtered = librosa.istft(filtered_stft)
    return filtered


def spectral_subtraction(noisy_audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Applies the spectral subtraction to the noisy signal."""
    
    # Magnitude spectrum
    noisy_stft = librosa.stft(noisy_audio)
    mag_noisy, phase_noisy = np.abs(noisy_stft), np.angle(noisy_stft)

    # Estimate that the first 0.5 seconds of the signal contains only noise
    noise_estimate = noisy_audio[:int(0.5 * sr)]
    est_mag = np.abs(librosa.stft(noise_estimate))
    # Apply spectral subtraction
    magnitude_denoised = np.maximum(mag_noisy - est_mag.mean(axis=1, keepdims=True), 0)

    # Reconstruct denoised signal using original phase
    stft_denoised = magnitude_denoised * np.exp(1j * phase_noisy)
    clean_audio = librosa.istft(stft_denoised)
    return clean_audio


def apply_denoising(output_dir: str, test_dataset: SpeechTestDataset, n: int = 10, sr: int = 48000):
    """Apply the denoising model to the test dataset.
    Save the denoised audio files to the disk.
    """
    for i in range(n):
        noisy, clean = test_dataset[i]
        spectral = spectral_subtraction(noisy)
        wiener = wiener_filtering(noisy, clean)
        os.makedirs(output_dir, exist_ok=True)
        sf.write(f"{output_dir}/spectral_{i}.wav", spectral, sr)
        sf.write(f"{output_dir}/wiener_{i}.wav", wiener, sr)
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
