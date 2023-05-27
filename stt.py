import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models


DATA_DIR = "data/mini_speech_commands"
SAMPLE_RATE = 16000  # Desired sample rate for the audio
NUM_MFCC = 13  # Number of MFCC coefficients to extract
AUDIO_LENGTH = 16000  # Desired fixed length for the audio



#1. Data preparation
commands = np.array(tf.io.gfile.listdir(str(DATA_DIR)))
commands = commands[(commands != 'READ.md') & (commands != '.DS_Store')]
print('Commands: ', commands)

trainDataset, valDataset = tf.keras.utils.audio_dataset_from_directory(
    directory=DATA_DIR,
    batch_size=64,
    validation_split=0.2, 
    seed=0,
    output_sequence_length=16000,
    subset='both'
)

labelNames = np.array(trainDataset.class_names)
print()
print('Label Names: ', labelNames)

trainDataset.element_spec

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels
  #Using 'squeeze' function to drop extra axis from data

trainDataset = trainDataset.map(squeeze, tf.data.AUTOTUNE)
valDataset = valDataset.map(squeeze, tf.data.AUTOTUNE)

testDataset = valDataset.shard(num_shards=2, index=0)
valDataset = valDataset.shard(num_shards=2, index=1)
#splitting val-dataset into test and validation sets



#2. Wav to spectrogram

def to_spectrogram(signal):
    stft = tf.signal.stft(
        signal, frame_length=255, frame_step=128
    ) 

    spectrogram = tf.abs(stft)

    return spectrogram

def makeSpectrogramDataset(dataset):
    return dataset.map(
        map_func=lambda audio, label:(to_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

trainSpectrogramDS = makeSpectrogramDataset(trainDataset)
valSpectrogramDS = makeSpectrogramDataset(valDataset)
testSpectrogramDS = makeSpectrogramDataset(testDataset)

for exampleSpectrograms, exampleSpectLabels in trainSpectrogramDS.take(1):
  break

inputShape = exampleSpectrograms.shape[1:]
print('Input shape: ', inputShape)
numLabels = len(labelNames)



#3. Model

inputShape = exampleSpectrograms.shape[1:]
print('Input shape: ', inputShape)
numLabels = len(labelNames)

normLayer = layers.Normalization()
normLayer.adapt(data=trainSpectrogramDS.map(map_func=lambda spec, label: spec))

#Standard CNN model
model = models.Sequential([
  layers.Input(shape=inputShape),
  layers.Resizing(32, 32),
  normLayer,
  layers.Conv2D(32, 3, activation='relu'),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(numLabels),
])

model.summary()

'''
inputShape = exampleSpectrograms.shape[1:]
print('Input shape: ', inputShape)
numLabels = len(labelNames)
# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=trainSpectrogramDS.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=inputShape),
    # Downsample the input.
    layers.Resizing(32, 32, interpolation="bilinear"),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(numLabels),
])

model.summary()
'''


#4. Training

#trainSpectrogramDS = trainSpectrogramDS.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
#valSpectrogramDS = valSpectrogramDS.cache().prefetch(tf.data.AUTOTUNE)
#testSpectrogramDS = testSpectrogramDS.cache().prefetch(tf.data.AUTOTUNE)
