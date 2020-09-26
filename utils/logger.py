import logging

def resultLogger(filename):
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename)

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.DEBUG)

    return logger

if __name__ == "__main__":
    logger = logger("../config/results.log")
    for i in range(1, 11):
        logger.info(f"{i}th number printed")



