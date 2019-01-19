import json
import torch
import torchvision
from .base import CustomDataset
from .base import BaseDataLoader
from .base import MultiTaskDataLoader


class CIFAR100Loader(BaseDataLoader):
    def __init__(self, batch_size=128, type='train', shuffle=True, drop_last=False):
        super(CIFAR100Loader, self).__init__(batch_size, type, shuffle, drop_last)

        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._create_dataloaders()


    def _create_dataloaders(self):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )

        if self.type == 'train':
            dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                    download=True, transform=transform)
            num_data = len(dataset)
            index = list(range(num_data))
            sampler = torch.utils.data.sampler.SubsetRandomSampler(index[:45000])
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     sampler=sampler,
                                                     drop_last=self.drop_last)
        elif self.type == 'valid':
            dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                    download=True, transform=transform)
            num_data = len(dataset)
            index = list(range(num_data))
            sampler = torch.utils.data.sampler.SubsetRandomSampler(index[45000:])
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     sampler=sampler,
                                                     drop_last=self.drop_last)
        elif self.type == 'test':
            dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                    download=True, transform=transform)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
        else:
            raise ValueError('Unknown data type: {}'.format(type))

        with open('data_loader/CIFAR100_fine2coarse.json', 'r') as f:
            data_info = json.load(f)

        images = [[] for _ in range(20)]
        labels = [[] for _ in range(20)]

        for batch_images, batch_labels in dataloader:
            for i, l in zip(batch_images, batch_labels):
                images[data_info['task'][l]].append(i)
                labels[data_info['task'][l]].append(data_info['subclass'][l])

        self.dataloader = []
        for task_images, task_labels in zip(images, labels):
            dataset = CustomDataset(data=task_images, labels=task_labels)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
            self.dataloader.append(dataloader)


    def get_loader(self, task=None):
        if task is None:
            return MultiTaskDataLoader(self.dataloader)
        else:
            assert task in list(range(20)), 'Unknown loader: {}'.format(task)
            return self.dataloader[task]


    @property
    def image_size(self):
        return 32


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes(self):
        return [5 for _ in range(20)]
