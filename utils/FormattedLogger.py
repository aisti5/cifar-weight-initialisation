import logging
logging.basicConfig(level=logging.INFO)


class FormattedLogger(logging.Logger):
    def __init__(self, name):
        super(FormattedLogger, self).__init__(name)
        formatter = logging.Formatter('%(asctime)s - %(name)15s - %(levelname)7s: %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        self.setLevel(logging.INFO)
