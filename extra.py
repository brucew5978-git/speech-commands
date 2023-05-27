
'''
def load_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)  # Remove the channel dimension if present
    return audio

def resize_audio(audio, length):
    # Pad or truncate audio to the desired length
    audio_length = tf.shape(audio)[0]
    padding = tf.maximum(length - audio_length, 0)
    padded_audio = tf.pad(audio, [[0, padding]])
    resized_audio = tf.slice(padded_audio, [0], [length])
    return resized_audio
'''

'''   
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(exampleSpectrograms[i].numpy(), ax)
    ax.set_title(labelNames[exampleSpectLabels[i].numpy()])

plt.show()

 
def spectrogram_to_mfcc(spectrogram, stft):

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stft.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
    spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]

    return mfcc    


def preprocess_dataset(data_dir, sample_rate, n_mfcc, audio_length):
    file_paths = tf.io.gfile.glob(os.path.join(data_dir, "*/*.wav"))
    labels = sorted(os.listdir(data_dir))
    
    audio_data = []
    label_data = []
    
    for file_path in file_paths:
        # Load audio file
        audio = load_audio(file_path)
        
        # Resize audio to a fixed length
        resized_audio = resize_audio(audio, audio_length)

        spectrogram, stft = to_spectrogram(resized_audio)
        
        
        # Preprocess audio and extract MFCCs
        mfccs = spectrogram_to_mfcc(spectrogram, stft)

        

        
        # Extract label from the directory name
        #label = tf.strings.split(tf.strings.split(file_path, os.path.sep)[-2], "_")[-1]
        
        #audio_data.append(mfccs)
        #label_data.append(label)
        
        
    
    return audio_data, label_data, labels


audio_data, label_data, labels = preprocess_dataset(data_dir, sample_rate, n_mfcc, audio_length)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()
'''