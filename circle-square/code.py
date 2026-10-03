import numpy as np
from PIL import Image, ImageOps
import tensorflow.keras

# Teachable Machine prints probabilities in scientific notation by default.
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model("keras_model.h5")

# One 224x224 RGB image, the Teachable Machine input size.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("test_photo.png")
image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
image_array = np.asarray(image)
image.show()

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)
print(prediction)
