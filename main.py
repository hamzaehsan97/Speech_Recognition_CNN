from audio import *
from predict import *
import sounddevice as sd
from scipy.io.wavfile import write


def testRecord():
    print("Recording for two seconds")    
    fs = 44100  # Sample rate
    seconds = 2  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)
    predictRecording()

    
print("PRESS RETURN TO RECORD")
key = input("PRESS RETURN TO RECORD :")
if key == "":
        testRecord()


