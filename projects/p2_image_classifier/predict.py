import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image

## Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type = str)
parser.add_argument('model_path', type = str)
parser.add_argument(
    '--top_k',
    type = int,
    default = 5
)
parser.add_argument(
    '--category_names',
    type = str
)

args = parser.parse_args()

## Load model and image

model = tf.keras.models.load_model(
    args.model_path,
    custom_objects = { 'KerasLayer': hub.KerasLayer },
    compile = False
)

# Copied from part 1
def process_image(image):
    image_tf = tf.convert_to_tensor(image)

    image_tf = tf.cast(image_tf, tf.float32)
    image_tf = tf.image.resize(image_tf, (224, 224))
    image_tf /= 255
    
    return image_tf.numpy()

image = np.asarray(Image.open(args.image_path))
image = process_image(image)
image = np.expand_dims(image, axis = 0)

## Predict

probs = model.predict(image).flatten()
preds = [(i + 1, p) for i, p in enumerate(probs)]
preds.sort(key = lambda k: k[1], reverse = True)

# Load name map
names = json.load(open(args.category_names))

# Results
print()
for pred in preds[:args.top_k]:
    print(f'{names[str(pred[0])]}: {pred[1]}')