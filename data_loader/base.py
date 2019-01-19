import numpy as np
import torch


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BaseDataLoader:
    def __init__(self, batch_size=1, type='train', shuffle=True, drop_last=False):
        pass

    def get_loader(self, task):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


class MultiTaskDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        self.num_tasks = len(self.dataloaders)
        self.task_order = list(range(self.num_tasks))
        self.size = max([len(d) for d in self.dataloaders]) * self.num_tasks

        self.task_step = 0
        self.data_step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.data_step >= self.size:
            self.data_step = 0
            raise StopIteration

        if self.task_step >= self.num_tasks:
            np.random.shuffle(self.task_order)
            self.task_step = 0

        task = self.task_order[self.task_step]

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.task_step += 1
        self.data_step += 1

        return data, labels, task