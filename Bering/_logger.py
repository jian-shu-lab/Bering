import os
import logging 
import datetime

def LOGGING():
    if not os.path.isdir('logging'): {os.mkdir('logging')}
    logging.basicConfig(
        filemode = 'w+', 
        level = logging.INFO, 
        format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s',
        filename = 'logging/logging_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log',
    )
    logger = logging.getLogger(__name__)
    return logger