import numpy as np
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path

from utils import get_files_from_dir, get_audio_data, split_audio


class SpeechTrainDataset(Dataset):

    def __init__(self, root_dir: Union[str, Path], return_paths: bool = False) -> None:
        """Pytorch Dataset class for training samples."""

        self.root_dir = Path(root_dir) / 'dataset'
        self.return_paths = return_paths

        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""

        print("\nLoading training data...")

        noisy_train_dir = self.root_dir / 'noisy_trainset_28spk_wav'
        clean_train_dir = self.root_dir / 'clean_trainset_28spk_wav'

        self.noisy_train_files = get_files_from_dir(noisy_train_dir)
        self.clean_train_files = get_files_from_dir(clean_train_dir)

        self.noisy_train = []
        self.clean_train = []

        if not self.return_paths:
            [self.noisy_train.extend(split_audio(*get_audio_data(f))) for f in self.noisy_train_files]
            [self.clean_train.extend(split_audio(*get_audio_data(f))) for f in self.clean_train_files]

            del self.noisy_train_files
            del self.clean_train_files

        print("Done!\n")


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

    def __init__(self, root_dir: Union[str, Path], return_paths: bool = False) -> None:
        """Pytorch Dataset class for testing samples. """
        self.root_dir = Path(root_dir) / 'dataset'
        self.return_paths = return_paths

        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""

        # Load the test data
        noisy_test_dir = self.root_dir / 'noisy_testset_wav'
        clean_test_dir = self.root_dir / 'clean_testset_wav'


        self.noisy_test_files = get_files_from_dir(noisy_test_dir)
        self.clean_test_files = get_files_from_dir(clean_test_dir)

        self.noisy_test = []
        self.clean_test = []

        if not self.return_paths:
            for file in self.noisy_test_files:
                audio, sr = get_audio_data(file)
                if len(audio) < sr:
                    print(f"Skipping {file}")
                    continue
                else:
                    self.noisy_test.append(audio)
            
            for file in self.clean_test_files:
                audio, sr = get_audio_data(file)
                if len(audio) < sr:
                    print(f"Skipping {file}")
                    continue
                else:
                    self.clean_test.append(audio)

            del self.noisy_test_files
            del self.clean_test_files

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        if self.return_paths:
            return len(self.noisy_test_files)
        else:
            return len(self.noisy_test)

    def __getitem__(self, idx):
        """Returns the item at index `idx`."""

        if self.return_paths:
            return self.noisy_test_files[idx], self.clean_test_files[idx]
        else:
            return self.noisy_test[idx], self.clean_test[idx]


def test_data_handling():

    # Test SpeechTrainDataset
    train_dataset = SpeechTrainDataset(root_dir='.', return_paths=False)
    print(f"Train dataset length: {len(train_dataset)}")
    noisy_data, clean_data = train_dataset[0]
    print(f"First training sample noisy shape: {noisy_data}, clean shape: {clean_data}")

    # Test SpeechTestDataset
    test_dataset = SpeechTestDataset(root_dir='.', return_paths=False)
    print(f"Test dataset length: {len(test_dataset)}")
    noisy_data, clean_data = test_dataset[0]
    print(f"First test sample noisy shape: {noisy_data.shape}, clean shape: {clean_data.shape}")
    print(f"Type of the first test sample noisy: {type(noisy_data)}, clean: {type(clean_data)}")

if __name__ == "__main__":
    test_data_handling()