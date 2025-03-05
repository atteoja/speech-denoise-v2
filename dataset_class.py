import numpy as np
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path

from utils import get_files_from_dir, get_audio_data, split_audio


class SpeechTrainDataset(Dataset):
<<<<<<< HEAD
    def __init__(self, root_dir: Union[str, Path], sr: int = 44500, features: bool = True) -> None:
        """Pytorch Dataset class for training samples."""

        self.root_dir = Path(root_dir) / 'dataset'
        self.sr = sr
        self.features = features
        self.noisy_train = []
        self.clean_train = []
=======
    def __init__(self, root_dir: Union[str, Path], return_paths: bool = False) -> None:
        """Pytorch Dataset class for training samples."""

        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
>>>>>>> origin/main
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
<<<<<<< HEAD

        print("Loading training data...")

        if self.features:
            noisy_train_dir = self.root_dir / 'noisy_trainset_28spk_wav_features'
            clean_train_dir = self.root_dir / 'clean_trainset_28spk_wav_features'
=======
>>>>>>> origin/main

        noisy_train_dir = self.root_dir / 'noisy_trainset_28spk_wav'
        clean_train_dir = self.root_dir / 'clean_trainset_28spk_wav'

        self.noisy_train_files = get_files_from_dir(noisy_train_dir)
        self.clean_train_files = get_files_from_dir(clean_train_dir)

        self.noisy_train = []
        self.clean_train = []

<<<<<<< HEAD
            for file in noisy_train_files:
                data, sr = get_audio_data(file, self.sr)
                segments = split_audio(data, sr)
                self.noisy_train.extend(segments)

            for file in clean_train_files:
                data, sr = get_audio_data(file, self.sr)
                segments = split_audio(data, sr)
                self.clean_train.extend(segments)

            print("Done!\n")
=======
        if not self.return_paths:
            [self.noisy_train.extend(split_audio(*get_audio_data(f))) for f in self.noisy_train_files]
            [self.clean_train.extend(split_audio(*get_audio_data(f))) for f in self.clean_train_files]

            del self.noisy_train_files
            del self.clean_train_files
>>>>>>> origin/main

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.noisy_train_files) if self.return_paths else len(self.noisy_train)

    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        if self.return_paths:
            return self.noisy_train_files[idx], self.clean_train_files[idx]
        else:
            return self.noisy_train[idx], self.clean_train[idx]

class SpeechTestDataset(Dataset):
<<<<<<< HEAD
    def __init__(self, root_dir: Union[str, Path], sr: int = 44500, features: bool = True) -> None:
        """Pytorch Dataset class for testing samples. """
        self.root_dir = Path(root_dir) / 'data'
        self.sr = sr
        self.features = features
        self.noisy_test = []
        self.clean_test = []
=======
    def __init__(self, root_dir: Union[str, Path], return_paths: bool = False) -> None:
        """Pytorch Dataset class for testing samples. """
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
>>>>>>> origin/main
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
<<<<<<< HEAD

        print("Loading testing data...")

        if self.features:
            noisy_test_dir = self.root_dir / 'noisy_testset_wav_features'
            clean_test_dir = self.root_dir / 'clean_testset_wav_features'
=======
        # Load the test data
        noisy_test_dir = self.root_dir / 'noisy_testset_wav'
        clean_test_dir = self.root_dir / 'clean_testset_wav'
>>>>>>> origin/main

        self.noisy_test_files = get_files_from_dir(noisy_test_dir)
        self.clean_test_files = get_files_from_dir(clean_test_dir)
   
        self.noisy_test = []
        self.clean_test = []

        if not self.return_paths:
            # Split the audio files into segments and extend the lists
            [self.noisy_test.extend(get_audio_data(f)) for f in self.noisy_test_files]
            [self.clean_test.extend(get_audio_data(f)) for f in self.clean_test_files]

            del self.noisy_test_files
            del self.clean_test_files

<<<<<<< HEAD
            noisy_test_files = get_files_from_dir(noisy_test_dir)
            clean_test_files = get_files_from_dir(clean_test_dir)

            self.noisy_test = [get_audio_data(f, sr=self.sr)[0] for f in noisy_test_files]
            self.clean_test = [get_audio_data(f, sr=self.sr)[0] for f in clean_test_files]

        print("Done!\n")


=======
>>>>>>> origin/main

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        if self.return_paths:
            return len(self.noisy_test_files)
        else:
            return len(self.noisy_test)
    
    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
<<<<<<< HEAD
        return self.noisy_test[idx], self.clean_test[idx]
    
def split_audio(audio_data, sr):

    segments = []
    time = int(np.floor(len(audio_data) / sr))

    for i in range(time):
        segment = audio_data[i*sr:(i+1)*sr]
        segments.append(segment)

    return segments

=======
        if self.return_paths:
            return self.noisy_test_files[idx], self.clean_test_files[idx]
        else:
            return self.noisy_test[idx], self.clean_test[idx]
>>>>>>> origin/main

def test_data_handling():
    
    # Test SpeechTrainDataset
    train_dataset = SpeechTrainDataset(root_dir='dataset/', return_paths=True)
    print(f"Train dataset length: {len(train_dataset)}")
    noisy_data, clean_data = train_dataset[0]
    print(f"First training sample noisy shape: {noisy_data}, clean shape: {clean_data}")
    
    # Test SpeechTestDataset
    test_dataset = SpeechTestDataset(root_dir='dataset/', return_paths=False)
    print(f"Test dataset length: {len(test_dataset)}")
    noisy_data, clean_data = test_dataset[0]
    print(f"First test sample noisy shape: {noisy_data.shape}, clean shape: {clean_data.shape}")
    print(f"Type of the first test sample noisy: {type(noisy_data)}, clean: {type(clean_data)}")

if __name__ == "__main__":
    test_data_handling()

