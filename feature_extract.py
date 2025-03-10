import pathlib
import os
import numpy as np
import librosa
from typing import List, Optional

from utils import get_audio_files, get_audio_data


def extract_mel_spectrogram(audio_signal: np.ndarray,
                        sr: int = 44500,
                        n_mels: int = 64,
                        n_fft: Optional[int] = 2048,
                        hop_length: Optional[int] = 512,
                        window: Optional[str] = 'hann') \
        -> np.ndarray: # old 'hamm'
    """Extracts and returns the mel spectrogram from the `audio_signal` signal."""

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal,
                                                    sr=sr,
                                                    n_mels=n_mels,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    window=window,
                                                    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram
    


def split_spectrum(audio_data: np.ndarray, sr: int = 44500, overlap: float = 0.5) -> List[np.ndarray]:
    """Splits the mel spectrogram into overlapping segments."""
    segment_length = sr
    step = int(segment_length * (1 - overlap))  # Overlap by 50%
    
    segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data) - segment_length, step)]
    return segments


def main(dataset_root: pathlib.Path):
    splits = ["trainset_28spk_wav", "testset_wav"]
    prefixes = ["clean_", "noisy_"]

    for prefix in prefixes:
        for split in splits:
            dataset_path = dataset_root / f"{prefix}{split}"

            if dataset_path.exists() and dataset_path.is_dir():
                audio_paths = get_audio_files(dataset_path)
                output_dir = dataset_root / f"{prefix}{split}_features"

                if output_dir.exists():
                    for file in output_dir.iterdir():
                        file.unlink()
                else:
                    os.mkdir(output_dir)
                
                print(f'Extracting features from {dataset_path} to {output_dir}')
                #mel_specs = []
                for audio_file in audio_paths:
                    y, sr = get_audio_data(audio_file)

                    if split == "testset_wav":
                        mel_spec = extract_mel_spectrogram(y, sr)
                        file_name = f"{audio_file.stem}.npy"
                        np.save(output_dir / file_name, mel_spec)
                        continue
                    
                    segments = split_spectrum(y)
                    for i, segment in enumerate(segments):
                        file_name = f"{audio_file.stem}_{i}.npy"
                        mel_spec = extract_mel_spectrogram(segment, sr)
                        np.save(output_dir / file_name, mel_spec)

            else:
                print(f"Skipping missing directory: {dataset_path}")

if __name__ == '__main__':
    dataset_root_path = pathlib.Path("dataset/") 
    main(dataset_root_path)