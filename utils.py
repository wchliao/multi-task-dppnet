import numpy as np


def permutations(num_new_items, inputs=None):
    if inputs is None:
        inputs = np.array([[]], dtype=np.int)

    out = []
    for input in inputs:
        for i in range(num_new_items):
            out.append(np.append(input, i))

    return np.array(out)


def _pareto_dominate(pa, pb):
    for a, b in zip(pa, pb):
        if a < b:
            return False
    return True


def _pareto_front(points):
    pareto_points = set()
    dominated_points = set()

    while len(points) > 0:
        point = points.pop()
        pareto = True
        remove_points = []

        for p in points:
            if _pareto_dominate(point, p):
                remove_points.append(p)
            elif pareto and _pareto_dominate(p, point):
                pareto = False

        if pareto:
            pareto_points.add(point)
        else:
            dominated_points.add(point)

        for p in remove_points:
            points.remove(p)
            dominated_points.add(p)

    pareto_points = [tuple(p) for p in pareto_points]
    dominated_points = [tuple(p) for p in dominated_points]

    return pareto_points, dominated_points


def pareto_front(points, num=None):
    if num is None:
        num = len(points)

    results = []
    tmp_points = points.copy()

    while len(results) < num:
        pareto_points, tmp_points = _pareto_front(tmp_points)
        results += pareto_points

    idx = []
    for p in results:
        idx.append(points.index(p))

    return results[:num], idx[:num]


def _conv_size(in_channels, out_channels, kernel_size):
    return in_channels * out_channels * kernel_size * kernel_size + in_channels


def _depthwise_conv_size(in_channels, out_channels, kernel_size):
    depthwise_size = in_channels * kernel_size * kernel_size + in_channels
    pointwise_size = in_channels * out_channels + out_channels

    return depthwise_size + pointwise_size


def _batchnorm_size(num_channels):
    return num_channels * 4


def estimate_single_task_model_size(layers, architecture, num_channels):
    size = 0
    in_channels = num_channels

    for layer, args in zip(layers, architecture):
        out_channels = args.num_channels
        kernel_size = layer.kernel_size

        if layer.type == 'conv':
            size += _conv_size(in_channels, out_channels, kernel_size)
            size += _batchnorm_size(out_channels)
        elif layer.type == 'depthwise-conv':
            size += _depthwise_conv_size(in_channels, out_channels, kernel_size)
            size += _batchnorm_size(out_channels)

        in_channels = out_channels

    return size


def estimate_multi_task_model_size(layers, architecture, num_channels):
    size = 0
    in_channels = num_channels

    for layer, args in zip(layers, architecture):
        shares = layer.share
        layer = layer.layer
        out_channels = args.num_channels
        kernel_size = layer.kernel_size

        layer_size = 0
        if layer.type == 'conv':
            layer_size += _conv_size(in_channels, out_channels, kernel_size)
            layer_size += _batchnorm_size(out_channels)
        elif layer.type == 'depthwise-conv':
            layer_size += _depthwise_conv_size(in_channels, out_channels, kernel_size)
            layer_size += _batchnorm_size(out_channels)

        if sum(shares) > 0:
            size += layer_size

        for share in shares:
            if not share:
                size += layer_size

        in_channels = out_channels

    return size
