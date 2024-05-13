"""
This module contains custom exceptions
"""
from ai_compendium.io_models import InferenceOutput


class BusinessException(Exception):
    """
    Base class for custom exceptions
    """

    def __init__(self, value):
        super(BusinessException, self).__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)


def custom_except(message, status_code=500):
    """
    Custom exception message to be shown on API
    :param message: Custom message to be shown for specific error occurred
    :param status_code: Status code of api request
    :return: result: Error dictionary with status code and message
    """
    InferenceOutput.status_code = status_code
    InferenceOutput.message = message
    InferenceOutput.data = {}
    return InferenceOutput
