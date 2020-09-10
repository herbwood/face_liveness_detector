from data_loader.data_generator import DataLoader
from utils.utils import configInfo

def main():
    dl = DataLoader(config="config.json")
    trainX, testX, trainY, testY = dl.split()
    print(trainX)
    print(f'print {__name__}')

if __name__ == "__main__":
    main()