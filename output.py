from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

sample_img_name = "./COVID-19_Radiography_Dataset/Pneumonia/Viral Pneumonia-800.png"


image = load_img(sample_img_name, target_size=(224, 224))
img = img_to_array(image)
img = img.reshape((1, 224, 224, 3))

model = load_model("./best_model.h5")


result = model.predict(img)
result = np.argmax(result, axis=-1)

if result == 0:
    print("Normal")
elif result == 1:
    print("Viral Pneumonia")
else:
    print("COVID +")

