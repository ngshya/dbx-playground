import logging
import os


class CustomLogger(logging.Logger):
    """
    Custom logging class extending the built-in logging.Logger to provide
    consistent logging across the package with the ability to specify
    the source of each log message.

    Extends:
        logging.Logger
    """

    def __init__(self, name: str, level: int | None = None) -> None:
        """
        Initialize the custom logger.

        If the environment variable LOGGING_LEVEL is set, its value will be used
        as the logging level instead of the provided level argument.

        Args:
            name (str): The name or source for the logger (e.g., function/module name).
            level (int | None, optional): Logging level. If None, defaults to INFO
                or to the level specified by the LOGGING_LEVEL environment variable.
        """
        # Get logging level from environment variable if set, otherwise use provided level or INFO
        env_level_str: str | None = os.getenv("LOGGING_LEVEL")
        if env_level_str:
            env_level_str = env_level_str.upper()
            # Convert string level to logging module level (int)
            env_level: int = logging.getLevelName(env_level_str)
            # If getLevelName returns a string, that means invalid level, fallback to INFO
            if isinstance(env_level, str):
                env_level = logging.INFO
            effective_level: int = env_level
        else:
            effective_level = level if level is not None else logging.INFO

        super().__init__(name, effective_level)

        if not self.hasHandlers():
            # Create console handler with a specific format and set level
            ch = logging.StreamHandler()
            ch.setLevel(effective_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.addHandler(ch)
