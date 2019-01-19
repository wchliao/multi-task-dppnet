class BaseModel:
    def __init__(self, layers, architecture, task_info):
        pass

    def train(self, train_data, test_data, num_epochs, learning_rate, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass

    @property
    def size(self):
        raise NotImplementedError
