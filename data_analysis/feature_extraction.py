from madmom.features.chords import CNNChordFeatureProcessor
from madmom.features.chords import CRFChordRecognitionProcessor

if __name__ == '__main__':
    feature_process = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    file_name = 'test/test.wav'
    features = feature_process(file_name)
    result = decode(features)
    count = 0
    for item in result:
        start_time, end_time, des = item[0], item[1], item[2]
        if des != 'N':
            if end_time - start_time <2.0:
                count += 1
            else:
                count += int((end_time - start_time) / 2.0)
            print count, '  ', item, '  ', end_time - start_time, 's'
