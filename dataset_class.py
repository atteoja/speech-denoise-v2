import numpy as np
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path

from utils import get_files_from_dir, get_audio_data, split_audio, collate_fn


class SpeechTrainDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Pytorch Dataset class for training samples."""


        self.root_dir = Path(root_dir)
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""

        noisy_train_dir = self.root_dir / 'noisy_trainset_28spk_wav'
        clean_train_dir = self.root_dir / 'clean_trainset_28spk_wav'

        self.noisy_train_files = get_files_from_dir(noisy_train_dir)
        self.clean_train_files = get_files_from_dir(clean_train_dir)


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.noisy_train_files)

    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        return (split_audio(*get_audio_data(self.noisy_train_files[idx])), split_audio(*get_audio_data(self.clean_train_files[idx])))


class SpeechTestDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Pytorch Dataset class for testing samples. """

        self.root_dir = Path(root_dir)
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""
        # Load the test data
        noisy_test_dir = self.root_dir / 'noisy_testset_wav'
        clean_test_dir = self.root_dir / 'clean_testset_wav'

        self.noisy_test_files = get_files_from_dir(noisy_test_dir)
        self.clean_test_files = get_files_from_dir(clean_test_dir)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.noisy_test_files)

    
    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        return get_audio_data(self.noisy_test_files[idx])[0], get_audio_data(self.clean_test_files[idx])[0]


def test_data_handling():
    
    # Test SpeechTrainDataset
    train_dataset = SpeechTrainDataset(root_dir='dataset/')
    print(f"Train dataset length: {len(train_dataset)}")
    noisy_data, clean_data = train_dataset[0]
    print(f"First training sample noisy shape: {noisy_data[0].shape}, clean shape: {clean_data[0].shape}")
    
    from torch.utils.data import DataLoader

    # Test SpeechTestDataset
    test_dataset = SpeechTestDataset(root_dir='dataset/')
    print(f"Test dataset length: {len(test_dataset)}")
    noisy_data, clean_data = test_dataset[0]
    print(f"First test sample noisy shape: {noisy_data.shape}, clean shape: {clean_data.shape}")
    print(f"Type of the first test sample noisy: {type(noisy_data)}, clean: {type(clean_data)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    for i, (noisy, clean) in enumerate(train_loader):
        print(f"Batch {i}: Noisy shape: {noisy[0].shape}, Clean shape: {clean[0].shape}")
        print(f"len of segments in batch: {len(noisy)} {len(clean)}")
        if i == 3:
            break



if __name__ == "__main__":
    test_data_handling()

