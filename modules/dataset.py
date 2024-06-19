import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class CustomDataset:
    def __init__(self, root_dir, train_split=0.8, transform=None):
        self.root_dir = root_dir
        self.train_split = train_split
        self.transform = transform
        self.__setup()

    def __setup(self):
        # Define transformation pipeline
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Load dataset and split into train/validation sets
        dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        self.num_classes_ = len(dataset.classes)

        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    @property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    @property
    def val_loader(self):
        return DataLoader(self.val_dataset, batch_size=8, shuffle=False)

    @property
    def num_classes(self):
        return self.num_classes_
