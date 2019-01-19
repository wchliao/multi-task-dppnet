from .base import BaseController
from search_space import search_space


class SingleTaskController(BaseController):
    def __init__(self, architecture, task_info):
        super(SingleTaskController, self).__init__(multi_task=False,
                                                   architecture=architecture,
                                                   search_space=search_space,
                                                   task_info=task_info
                                                   )
