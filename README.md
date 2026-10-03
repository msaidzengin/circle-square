# circle-square

Teachable Machine image classifier that distinguishes circles from squares.

The Keras model, labels, test photo, and inference script were committed on 20 November 2020. `shapes.zip`, added the same day, holds the drawing set used with the project: circles, squares, and triangles. The exported model only has two classes, listed in `circle-square/labels.txt`: Circle and Square.

## Run

From the `circle-square` directory:

```bash
pip install tensorflow pillow numpy
python code.py
```

The script loads `keras_model.h5`, fits `test_photo.png` to 224×224, opens that resized image, and prints one probability per class. Index 0 is Circle and index 1 is Square.
