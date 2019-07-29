from audio import *
import sounddevice as sd
from scipy.io.wavfile import write
from pynput.keyboard import Key, Listener




def testRecord():
    fs = 44100  # Sample rate
    seconds = 1  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)
    funct()

# Process the output.wav file
# Use the output.wav file from the model

def on_press(Key):
    print("RECORDING FOR 2 SECONDS")
    listener.stop()
    testRecord()
    

with Listener(on_press=on_press) as listener:
    listener.join()