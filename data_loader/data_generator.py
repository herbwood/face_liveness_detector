from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import numpy 
from utils.utils import configInfo
import os


class DataLoader:
    def __init__(self, config="../config.json"):
        self.config = configInfo(config)
        self.data = []
        self.labels = []

        print("[INFO] loading images...")
        imagePaths = list(paths.list_images(self.config["dataset"]))


        for imagePath in imagePaths:
            # 파일 이름에서 클래스 레이블을 추출하고 이미지를 로드한 다음, 32x32 크기 조정  
            label = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (32, 32))

            # 데이터 및 라벨 목록을 각각 업데이트
            self.data.append(image)
            self.labels.append(label)

        self.data = imagePreprocess(self.data)
        self.label = labelPreprocess(self.labels)
        


    def imagePreprocess(self, data):
        return np.array(data, dtype="float") / 255.0
    
    def labelPreprocess(self, labels):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        labels = np_utils.to_categorical(labels, 2)

        return labels

    def split(self, test_size=0.25):
        (trainX, testX, trainY, testY) = train_test_split(self.data, self.labels, test_size=test_size, random_state=42)
        return trainX, testX, trainY, testY

if __name__ == "__main__":
    dl = DataLoader()
    trainX, testX, trainY, testY = dl.split()
    print(trainX)