import yaml
from collections import namedtuple


# Named tuples for configurations

with open('configs/train.yaml', 'r') as f:
    _configs = yaml.load(f)

AgentConfigs = namedtuple('AgentConfigs', _configs['agent'].keys())
PredictorConfigs = namedtuple('PredictorConfigs', _configs['predictor'].keys())
ModelConfigs = namedtuple('ModelConfigs', _configs['model'].keys())
Configs = namedtuple('Configs', ['agent', 'predictor', 'model'])

with open('configs/architecture.yaml', 'r') as f:
    _configs = yaml.load(f)

LayerArguments = namedtuple('LayerArguments', _configs[0].keys())

with open('configs/search_space.yaml', 'r') as f:
    _configs = yaml.load(f)

Layer = namedtuple('Layer', _configs[0].keys())
ShareLayer = namedtuple('ShareLayer', ['layer', 'share'])


# Others

TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels', 'num_tasks'])
