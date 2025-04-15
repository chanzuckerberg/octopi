from torch.utils.data import Dataset
from monai.transforms import Compose

class DynamicDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def update_data(self, new_data):
        """Update the internal dataset with new data."""
        self.data = new_data