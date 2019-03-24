from datetime import datetime
import os
import logging

def create_logging():
    files = os.listdir('../log')
    for f in files:
        if f.endswith('.log'):
            os.remove('../log/' + f)

    # get tf logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler('../log/'+datetime.now().strftime('%Y-%m-%d %H-%M-%S')+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)