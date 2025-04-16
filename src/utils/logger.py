import logging
import os


class PipelineLogger(logging.Logger):
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Configure handlers based on parameters
        self._setup_handlers(log_file, level)

    def _setup_handlers(self, log_file, level):
        c_handler = logging.StreamHandler()
        c_handler.setLevel(level)
        c_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            f_handler = logging.FileHandler(log_file)
            f_handler.setLevel(level)
            f_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            f_handler.setFormatter(f_format)
            self.logger.addHandler(f_handler)

        self.logger.addHandler(c_handler)
