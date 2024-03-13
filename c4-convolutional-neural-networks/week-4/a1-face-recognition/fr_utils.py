import numpy as np
import tensorflow as tf


def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    embedding = model.predict(np.expand_dims(img, axis=0), verbose=0)
    return embedding / np.linalg.norm(embedding, ord=2)


persons = ['andrew', 'arnaud', 'benoit', 'bertrand', 'dan', 'danielle',
           'felix', 'kevin', 'kian', 'sebastiano', 'tian', 'younes']

model = tf.keras.models.load_model('facenet')

database = {person: img_to_encoding(f'faces/{person}.jpg', model) for person in persons}
