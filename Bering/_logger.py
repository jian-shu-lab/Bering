import os
import logging 
import datetime
import numpy as np

def LOGGING():
    if not os.path.isdir('logging'):
        os.mkdir('logging')
    random_number = np.random.randint(1000, 2000)
    logging.basicConfig(
        filemode = 'w+', 
        level = logging.INFO, 
        format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s',
        filename = 'logging/logging_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(random_number) + '.log'
    )
    logger = logging.getLogger(__name__)
    return logger