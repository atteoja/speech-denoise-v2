from typing import List, Union, Tuple
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa
import torch


def get_files_from_dir(dir_name: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    
    """Returns the files in the directory `dir_name` using the pathlib package."""
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_files(dir_name: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`."""

    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-4:] == '.wav']


def get_audio_data(audio_file: Union[str, pathlib.Path], sr: int = 44500) -> Tuple[np.ndarray, float]:
    """Loads and returns the audio data from the `audio_file`. """

    return librosa.core.load(path=audio_file, sr=16000, mono=True) # mono=True


def split_audio(audio_data: np.ndarray, sr: int, overlap: float = 0.25) -> List[np.ndarray]:
    """Splits the audio data into overlapping segments."""
    segment_length = sr * 2  # 2 seconds
    step = int(sr * overlap)  # 0.5 seconds overlap
    segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data) - segment_length, step)]
    
    # Handle the last segment 
    remaining_samples = len(audio_data) % segment_length
    if remaining_samples > 0:
        # remaining samples + padding
        last_segment = np.pad(audio_data[-remaining_samples:], (0, segment_length - remaining_samples), mode='constant')
        segments.append(last_segment)
    
    return segments



def collate_fn(batch):
    """
    Custom collate function for batching a list of tuples (noisy, clean) including list of segments.
    Flatten lists into a single lists of tensors
    """
    
    # Unzip batch into noisy and clean data
    input_seq, label_seq = zip(*batch)

    # Flatten the list of segments
    input_flat = [torch.tensor(segment) for input_item in input_seq for segment in input_item]
    label_flat = [torch.tensor(segment) for label_item in label_seq for segment in label_item]

    # Stack the individual segments into tensors
    inputs_batch = torch.stack(input_flat)
    labels_batch = torch.stack(label_flat)

    return inputs_batch, labels_batch


def test_utils():
    # Test get_files_from_dir_with_pathlib
    test_dir = "noisy_testset_wav"
    files = get_files_from_dir(test_dir)
    print(f"Files in {test_dir}: {files}")
    
    # Test get_audio_files_from_subdirs
    audio_files = get_audio_files(test_dir)
    print(f"Audio files in {test_dir} subdirectories: {audio_files}")
    
    # Test get_audio_file_data
    if audio_files:
        audio_data, sr = get_audio_data(audio_files[0])
        print(f"Audio data for {audio_files[0]}: {audio_data.shape}, Sampling rate: {sr}")

if __name__ == "__main__":
    test_utils()

