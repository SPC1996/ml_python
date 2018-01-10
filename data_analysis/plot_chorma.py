# encoding=utf-8
import matplotlib.pyplot as plt
import librosa.display


def draw(wav_path, type='stft', sr=22050, start=0.0, duration=None):
    y, sr = librosa.load(wav_path, sr=sr, offset=start, duration=duration)
    if type == 'stft':
        chroma_data = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    elif type == 'cqt':
        chroma_data = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    elif type == 'cens':
        chroma_data = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
    else:
        chroma_data = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    plt.figure()
    plt.subplot(1, 1, 1)
    librosa.display.specshow(chroma_data, y_axis='chroma', x_axis='time')
    plt.title('chroma_data_' + type)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    draw('test/A#05.wav', type='cqt', start=0.0, duration=10)