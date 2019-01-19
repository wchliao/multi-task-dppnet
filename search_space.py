import yaml
from namedtuple import Layer

with open('configs/search_space.yaml', 'r') as f:
    configs = yaml.load(f)

search_space = [Layer(**layer) for layer in configs]