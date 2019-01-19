import argparse
import yaml
from namedtuple import TaskInfo, AgentConfigs, PredictorConfigs, ModelConfigs, Configs, LayerArguments
from data_loader import CIFAR100Loader
from controller import SingleTaskController


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--type', type=int, default=1, help='1: Single task experiment \n')
    parser.add_argument('--data', type=int, default=1, help='1: CIFAR-100')
    parser.add_argument('--task', type=int, default=None)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--path', type=str, default='saved_models/default/')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    configs = _load_configs()
    architecture = _load_architecture()

    if args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='train', drop_last=True)
        valid_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='valid', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels,
                             num_tasks=1
                             )

        train_data = train_data.get_loader(args.task)
        valid_data = valid_data.get_loader(args.task)

        agent = SingleTaskController(architecture=architecture, task_info=task_info)

    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.train(train_data=train_data,
                valid_data=valid_data,
                configs=configs,
                save=args.save,
                path=args.path,
                verbose=args.verbose
                )


def evaluate(args):
    configs = _load_configs()
    architecture = _load_architecture()

    if args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='train', drop_last=True)
        test_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels,
                             num_tasks=1
                             )

        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)

        agent = SingleTaskController(architecture=architecture, task_info=task_info)

    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    best_architectures, results = agent.eval(train_data=train_data,
                                             test_data=test_data,
                                             configs=configs,
                                             save=args.save,
                                             path=args.path
                                             )

    for architecture, result in zip(best_architectures, results):
        print('Architecture:')
        for layer in architecture:
            print(layer)
        for key, value in result.items():
            print('{}: {}'.format(key, value))
        print()

def _load_configs():
    with open('configs/train.yaml', 'r') as f:
        configs = yaml.load(f)

    agent_configs = AgentConfigs(**configs['agent'])
    predictor_configs = PredictorConfigs(**configs['predictor'])
    model_configs = ModelConfigs(**configs['model'])

    return Configs(agent=agent_configs, predictor=predictor_configs, model=model_configs)


def _load_architecture():
    with open('configs/architecture.yaml', 'r') as f:
        configs = yaml.load(f)

    return [LayerArguments(**config) for config in configs]


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    main()