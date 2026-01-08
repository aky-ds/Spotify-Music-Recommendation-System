import sys
from src.logger.logger import logging
class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message

        # Get exception info
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            self.line_no = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.line_no = None
            self.file_name = None

    def __str__(self):
        return f"The error is '{self.error_message}', in the file '{self.file_name}', at line {self.line_no}"


if __name__ == '__main__':
    try:
        logging.info('Logging have been started')
        a = 20 / 0
    except Exception as e:
        raise CustomException(e)
