"""Logger rewritten by Ziwen Zhuang. (from rllab logging) """
from exptools.logging._logger import Logger

import warnings

# Referring to https://stackoverflow.com/questions/9942536/how-to-fake-proxy-a-class-in-python
# If you find any better solution to the needs please email me at zhuangzw@shanghaitech.edu.cn
""" 
* code need:
    - In the exptools, I wish the users can access the global logger after initialization the logging
    system simply by `from exptools.logging import logger`
    - There should be a global switch to activate/deactivate this global logger
    - Also, I wish the logger being implmennted by a class such that the users can instantiate a
    Logger for them when they want.

* current solution:
    - As you see, I implemented a `Proxy` class to proxy the real `Logger`
    - Also, the activate/deactivate switch is done using context module.
"""
WARN_STR = "You have not initialized exptools logger. Please use log_context in exptools.logging.context.logger_context"
class Proxy:
    def set_client(self, client):
        self._client = client
    def unset_client(self):
        del self._client
    """ NOTE: don' implement __getattribute__ to prevent unlimited recursion
    refer to https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
    """
    def __getattr__(self, name: str):
        if hasattr(self, "_client"):
            return getattr(self._client, name)
        warnings.warn(WARN_STR)

logger = Proxy()
