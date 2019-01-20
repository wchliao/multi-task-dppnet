from .base import BaseController
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskControllerFull(BaseController):
    def __init__(self, architecture, task_info):
        search_space_separate = [ShareLayer(layer=layer, share=False) for layer in layers]
        search_space_share = [ShareLayer(layer=layer, share=True) for layer in layers]
        search_space = search_space_separate + search_space_share

        super(MultiTaskControllerFull, self).__init__(multi_task=True,
                                                      architecture=architecture,
                                                      search_space=search_space,
                                                      task_info=task_info
                                                      )
