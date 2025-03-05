from typing import List, Union, Tuple
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa


def get_files_from_dir(dir_name: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    
    """Returns the files in the directory `dir_name` using the pathlib package."""
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_files(dir_name: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`."""

    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-4:] == '.wav']


def get_audio_data(audio_file: Union[str, pathlib.Path]) -> Tuple[np.ndarray, float]:
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

