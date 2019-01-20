import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from .base import BaseModel
from .core import SingleTaskCoreModel


class SingleTaskModel(BaseModel):
    def __init__(self, layers, architecture, task_info):
        super(SingleTaskModel, self).__init__(layers, architecture, task_info)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SingleTaskCoreModel(layers=layers, architecture=architecture, task_info=task_info)
        self.model = nn.DataParallel(self.model).to(self.device)


    def train(self,
              train_data,
              valid_data,
              num_epochs=20,
              learning_rate=0.1,
              save_history=False,
              path='saved_models/default/',
              verbose=False
              ):

        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        accuracy = []

        for epoch in range(num_epochs):
            scheduler.step()
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(valid_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, path)

        return accuracy[-1]


    def _save_history(self, history, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, data):
        correct = 0
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            self.model.train()

            return correct / total


    def save(self, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'model')

        torch.save(self.model.state_dict(), filename)


    def load(self, path='saved_models/default/'):
        if os.path.isdir(path):
            filename = os.path.join(path, 'model')
            self.model.load_state_dict(torch.load(filename))


    @property
    def size(self):
        size = 0
        for t in self.model.state_dict().values():
            size += np.prod(t.size())

        return size
