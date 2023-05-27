
#pip install -U -q tensorflow tensorflow_datasets
#apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
#Source: https://www.tensorflow.org/tutorials/audio/simple_audio

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'data/mini_speech_commands'

#Downloading "speech commands" dataset
dataDir = pathlib.Path(DATASET_PATH)
if not dataDir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')
  

commands = np.array(tf.io.gfile.listdir(str(dataDir)))
commands = commands[(commands != 'READ.md') & (commands != '.DS_Store')]
print('Commands: ', commands)

trainDataset, valDataset = tf.keras.utils.audio_dataset_from_directory(
    directory=dataDir,
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


#Visually displaying wav data
for exampleAudio, exampleLabels in trainDataset.take(1):
  print(exampleAudio.shape)
  print(exampleLabels.shape)

labelNames[[1,1,3,0]]

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
  if i>=n:
    break
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(exampleAudio[i].numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = labelNames[exampleLabels[i]]
  ax.set_title(label)
  ax.set_ylim([-1.1,1.1])

plt.show()



#1. Converting waveforms to spectrograms

def getSpectrogram(targetWaveform):
  #STFT = short-time Fourier transform
  spectrogram = tf.signal.stft(
      targetWaveform, frame_length=255, frame_step=128
  )

  #Magnitude of STFT
  spectrogram = tf.abs(spectrogram)

  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram



#2. Training preparation 
def makeSpectrogramDataset(dataset):
    return dataset.map(
        map_func=lambda audio, label:(getSpectrogram(audio), label),
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



#3. Training and evaluation set testing

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)
'''
EPOCHS = 10
history = model.fit(
  trainSpectrogramDS,
  validation_data=valSpectrogramDS,
  epochs=EPOCHS,
  callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
)

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

#Evaluating model accuracy 
model.evaluate(testSpectrogramDS, return_dict=True)
'''