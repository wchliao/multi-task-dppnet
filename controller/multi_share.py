from .base import BaseController
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskControllerShare(BaseController):
    def __init__(self, architecture, task_info):
        search_space = [ShareLayer(layer=layer, share=True) for layer in layers]

        super(MultiTaskControllerShare, self).__init__(multi_task=True,
                                                       architecture=architecture,
                                                       search_space=search_space,
                                                       task_info=task_info
                                                       )
