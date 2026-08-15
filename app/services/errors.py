"""Domain exceptions with user-safe Japanese messages."""


class AppError(Exception):
    """An expected error safe to show in the UI."""


class InvalidSourceError(AppError):
    pass


class DownloadError(AppError):
    pass


class DependencyError(AppError):
    pass


class ProcessingError(AppError):
    pass
