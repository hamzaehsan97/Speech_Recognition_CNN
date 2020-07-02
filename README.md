# Speech Recognition using Convolutional Neural Networks

This project uses the keras library to build a convolutional neural network based speech recognition model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

1) Make sure python 3 is installed on your device before going to next steps.
2) Project includes a requirements.txt file. In terminal got 

### Installing

A step by step series of examples that tell you how to get a development env running
```
1) Clone the project onto your local device.
2) In terminal/cmd line, go to project directory and run 'wandb login'
```
3 a) To use the pretrained model use the following steps
```
1) Run "wandb run python record.py"
2) Go to project directory, add that file to sub folder #Prediction in the folder #Predict
3) In terminal/cmd run "rm .DS_Store" in both folders
4) Run main.py
```
3 b) To train model on your data use the following steps
```
1) Collect recordings of the words required in .WAV format in length roughly 2 seconds.
2) Add recordings under labeled sub-folders in the data folder.
3) Run "rm .DS_Store" in all subfolders
4) Run "wandb run python audio.py"
5) Follow steps to use pretrained model to use the new model.
```

## Running the tests

Adding any of the following lines to main.py or audio.py (after the model is fit) can help with understanding the model.

1) print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
2) print(model.evaluate(X_test,y_test_hot, verbose=0))


## Built With

* [Wandb](http://www.wandb.com/) - Model training visualization
* [Keras](https://www.Keras.com) - Machine Learning Library
* [TensorFlow](https://www.Tensorflow.com) - Machine Learning Library

## Contributing

1) Currently using os.rename in record.py corrupts the output.wav file and makes the testing procedure ineffecient. If you find a more convinient solution your help will be appreciated. :)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Lucas from https://www.youtube.com/channel/UCBp3w4DCEC64FZr4k9ROxig helped me understand many of the key concepts.
* 3Blue1Brown https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw has some great videos on the fundamentals of Neural Nets.
* Neil Mix
