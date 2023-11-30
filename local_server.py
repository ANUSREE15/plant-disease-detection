import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate('cardamom-gecka-firebase-adminsdk-pv62c-91769526a8.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://cardamom-gecka-default-rtdb.firebaseio.com/',
    'storageBucket': 'cardamom-gecka.appspot.com'
})

# Load the saved model from file

model = tf.keras.models.load_model("./ModelFile/cardamom.h5")

CLASSES = ['Healthy', 'LeafBlight', 'LeafSpot']

def predict():
    img_path = './Uploads/capture.jpg'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make a prediction on the preprocessed image
    preds = model.predict(x)
    predicted_class = np.argmax(preds)
    return(CLASSES[predicted_class])

def checkDatabase():
    ref = db.reference('/Mode')
    mode = ref.get()
    if(mode == '1'):
        image_file_name = 'capture.jpg'
        bucket = storage.bucket()
        image_blob = bucket.blob(image_file_name)
        image_blob.download_to_filename('./Uploads/capture.jpg')

        ref = db.reference('/Result')
        data = predict()
        ref.set(data)

        ref = db.reference('/Mode')
        data = 0
        ref.set(data)

while True:
    checkDatabase()
    print("Server Under Standby Mode")
