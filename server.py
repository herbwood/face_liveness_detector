import pyrebase
from utils.utils import configInfo
from flask import *

config = configInfo("config/config.json")
config = config["firebase_config"]

# firebase config of app initialize
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()


def stream_handler(message):
    print(message["event"]) # put
    print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}

    try:
        if message["data"].startswith("com.google.android.gms.tasks.zzu@"):
            updated_phone_number = message['path'].split('/')[1]
            storage.child(f"videos/{updated_phone_number}").download(f"./video/{updated_phone_number}.mp4")
            print("Downloaded")

    except:
        print('Exception occured')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    my_stream = db.child("UserList").stream(stream_handler)

if __name__ == "__main__":
    app.run(debug=True)