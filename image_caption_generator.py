import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add

# Dummy word index for demo
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = {'startseq':1, 'a':2, 'dog':3, 'is':4, 'playing':5, 'in':6, 'the':7, 'park':8, 'endseq':9}
index_word = {v: k for k, v in tokenizer.word_index.items()}
vocab_size = len(tokenizer.word_index) + 1
max_length = 10

# Encoder: InceptionV3
def build_cnn_encoder():
    base_model = InceptionV3(weights='imagenet')
    model = Model(base_model.input, base_model.layers[-2].output)
    return model

# Decoder: RNN
def define_decoder_model():
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    decoder1 = Add()([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

# Extract features from image
def extract_features(filename, encoder):
    img = image.load_img(filename, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = encoder.predict(x, verbose=0)
    return features

# Greedy caption generation
def generate_caption(model, photo):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Main runner
if __name__ == "__main__":
    encoder = build_cnn_encoder()
    decoder = define_decoder_model()

    image_path = "images/sample.jpg"
    if not os.path.exists(image_path):
        print("Please place an image at images/sample.jpg to test.")
        exit()

    photo = extract_features(image_path, encoder)
    caption = generate_caption(decoder, photo)
    print("Generated Caption:", caption)