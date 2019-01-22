import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import pickle
import utils
from model import SingleTaskModel, MultiTaskModel


class BaseController:
    def __init__(self, multi_task, architecture, search_space, task_info):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if multi_task:
            self.build_model = MultiTaskModel
        else:
            self.build_model = SingleTaskModel

        self.architecture = architecture
        self.task_info = task_info

        self.num_layers = len(architecture)

        convs = []
        non_convs = []

        if multi_task:
            for s in search_space:
                if 'conv' in s.layer.type:
                    convs.append(s)
                else:
                    non_convs.append(s)
        else:
            for s in search_space:
                if 'conv' in s.type:
                    convs.append(s)
                else:
                    non_convs.append(s)

        self.search_space = convs + non_convs

        self.search_size = []
        for i in range(self.num_layers):
            if i == 0:
                self.search_size.append(len(convs))
            elif architecture[i].num_channels != architecture[i-1].num_channels:
                self.search_size.append(len(convs))
            else:
                self.search_size.append(len(self.search_space))

        self.predictor = AccuracyPredictor(search_size=len(self.search_space),
                                           device=self.device
                                           )


    def train(self,
              train_data,
              valid_data,
              configs,
              save=True,
              path='saved_models/default',
              verbose=False
              ):

        best_architectures = utils.permutations(num_new_items=self.search_size[0])
        for i in range(1, self.num_layers):
            if len(best_architectures) * self.search_size[i] > configs.agent.num_models:
                break
            else:
                best_architectures = utils.permutations(num_new_items=self.search_size[i], inputs=best_architectures)

        start_num_layers = len(best_architectures[0])

        for num_layers in range(start_num_layers, self.num_layers + 1):

            # Prepare samples

            if verbose:
                print('Sampling {}-layer samples.'.format(num_layers))

            samples_path = os.path.join(path, '{}-layer-samples'.format(num_layers))
            if os.path.isdir(samples_path):
                samples, accs = self._load_samples(samples_path)

            else:
                samples = best_architectures
                accs = []

                for sample in samples:
                    layers = [self.search_space[ID] for ID in sample]
                    model = self.build_model(layers, self.architecture[:num_layers], self.task_info)
                    accuracy = model.train(train_data=train_data,
                                           valid_data=valid_data,
                                           num_epochs=configs.model.num_epochs,
                                           learning_rate=configs.model.learning_rate,
                                           save_history=False,
                                           verbose=False
                                           )
                    accs.append(accuracy)

                if save:
                    self._save_samples(samples, accs, samples_path)

            # Train accuracy predictor

            if verbose:
                print('Training predictor for {}-layer models.'.format(num_layers))

            self.predictor.update(samples=samples,
                                  accuracies=accs,
                                  num_epochs=configs.predictor.num_epochs,
                                  learning_rate=configs.predictor.learning_rate
                                  )

            # Mutate next layer

            if num_layers + 1 < self.num_layers:
                if verbose:
                    print('Mutating {}-layer samples.'.format(num_layers))

                best_architectures = utils.permutations(num_new_items=self.search_size[num_layers],
                                                        inputs=best_architectures)
                accs = self.predictor.predict(best_architectures).detach().cpu().numpy()
                accs_order = accs.argsort()[::-1]

                best_architectures = best_architectures[accs_order][:configs.agent.num_candidate_models]
                accs = accs[accs_order][:configs.agent.num_candidate_models]

                model_sizes = []
                for architecture in best_architectures:
                    layers = [self.search_space[ID] for ID in architecture]
                    model_sizes.append(utils.estimate_model_size(layers=layers,
                                                                 num_tasks=self.task_info.num_tasks,
                                                                 architecture=self.architecture[:num_layers + 1],
                                                                 num_channels=self.task_info.num_channels
                                                                 )
                                       )

                objectives = [(acc, -model_size) for (acc, model_size) in zip(accs, model_sizes)]
                _, idx = utils.pareto_front(objectives, num=configs.agent.num_models)
                best_architectures = best_architectures[idx]

        if save:
            if verbose:
                print('Training final models.')

            architectures = []
            accs = []
            model_sizes = []

            for architecture in best_architectures:
                layer_IDs = [self.search_space[ID] for ID in architecture]
                model = self.build_model(layer_IDs, self.architecture, self.task_info)
                accuracy = model.train(train_data=train_data,
                                       valid_data=valid_data,
                                       num_epochs=configs.model.num_epochs,
                                       learning_rate=configs.model.learning_rate,
                                       save_history=False,
                                       verbose=False
                                       )

                architectures.append(layer_IDs)
                accs.append(accuracy)
                model_sizes.append(model.size)

            self.save(architectures, accs, model_sizes, path)


    def _save_samples(self, samples, accs, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'samples.pkl'), 'wb') as f:
            pickle.dump(samples, f)
        with open(os.path.join(path, 'accs.json'), 'w') as f:
            json.dump(accs, f)


    def _load_samples(self, path):
        with open(os.path.join(path, 'samples.pkl'), 'rb') as f:
            samples = pickle.load(f)
        with open(os.path.join(path, 'accs.json'), 'r') as f:
            accs = json.load(f)

        return samples, accs


    def eval(self,
             train_data,
             test_data,
             configs,
             save=True,
             path='saved_models/default'
             ):

        architectures, _, _ = self.load(path)
        accs = []
        model_sizes = []

        for architecture in architectures:
            model = self.build_model(architecture, self.architecture, self.task_info)
            accuracy = model.train(train_data=train_data,
                                   valid_data=test_data,
                                   num_epochs=configs.model.num_epochs,
                                   learning_rate=configs.model.learning_rate,
                                   save_history=False,
                                   verbose=False
                                   )

            accs.append(accuracy)
            model_sizes.append(model.size)

        if save:
            self.save(architectures, accs, model_sizes, os.path.join(path, 'eval'))

        accs_order = np.argsort(accs)[::-1]

        architectures = np.array(architectures)[accs_order].tolist()
        accs = np.array(accs)[accs_order]
        model_sizes = np.array(model_sizes)[accs_order]

        results = [{'Accuracy': acc, 'Model size': model_size} for acc, model_size in zip(accs, model_sizes)]

        return architectures, results


    def save(self, architectures, accs, model_sizes, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'best_architectures.pkl'), 'wb') as f:
            pickle.dump(architectures, f)
        with open(os.path.join(path, 'best_architectures_accs.json'), 'w') as f:
            json.dump(accs, f)
        with open(os.path.join(path, 'best_architectures_model_sizes.json'), 'w') as f:
            json.dump(model_sizes, f)


    def load(self, path):
        with open(os.path.join(path, 'best_architectures.pkl'), 'rb') as f:
            architectures = pickle.load(f)
        with open(os.path.join(path, 'best_architectures_accs.json'), 'r') as f:
            accs = json.load(f)
        with open(os.path.join(path, 'best_architectures_model_sizes.json'), 'r') as f:
            model_sizes = json.load(f)

        return architectures, accs, model_sizes


class AccuracyPredictor(nn.Module):
    def __init__(self, search_size, hidden_size=128, device=torch.device('cpu')):
        super(AccuracyPredictor, self).__init__()

        self.embeddings = torch.nn.Embedding(search_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        self.search_size = search_size
        self.hidden_size = hidden_size
        self.device = device

        self.to(device)


    def forward(self, samples):
        inputs = self.embeddings(samples)
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs[-1])
        predictions = F.sigmoid(outputs)

        return torch.squeeze(predictions)


    def update(self, samples, accuracies, num_epochs=300, learning_rate=8e-3):
        self.train()

        samples_time = np.transpose(samples)
        samples_time = torch.tensor(samples_time, dtype=torch.long, device=self.device)
        accuracies = torch.tensor(accuracies, device=self.device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(num_epochs):
            predictions = self.forward(samples_time)
            loss = torch.abs(predictions - accuracies)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def predict(self, samples):
        self.eval()

        samples_time = np.transpose(samples)
        samples_time = torch.tensor(samples_time, dtype=torch.long, device=self.device)

        return self.forward(samples_time)
