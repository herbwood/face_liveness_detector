import pyrebase
from utils.utils import configInfo
import os

def download_from_firebase(phonenumber, firebase_dir="videos", firebase_path= "2020-09-22 17:24:330102345679",
                           config="../config/config.json"):
    config = configInfo(config)

    firebase = pyrebase.initialize_app(config["firebase_config"])
    storage = firebase.storage()
    db = firebase.database()
    company = db.child("UserList").get()

    bdict = company.val()

    download_dir = config["video_save_path"]

    for i in bdict.keys():
        print(bdict[i])
        if i == phonenumber:
            download_path = os.path.join(download_dir, f"{i}.mp4")
            storage.child(os.path.join(firebase_dir, firebase_path)).download(download_path)

if __name__ == "__main__":
    download_from_firebase("01012345678")



