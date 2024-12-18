from fastapi import status

from src.exceptions import DetailedHTTPException


class TaskNotFound(DetailedHTTPException):
    STATUS_CODE = status.HTTP_404_NOT_FOUND
    DETAIL = "Task not found"


class TaskSubmissionError(DetailedHTTPException):
    STATUS_CODE = status.HTTP_500_INTERNAL_SERVER_ERROR
    DETAIL = "Failed to submit the task. Please try again later."
