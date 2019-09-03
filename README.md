# speechRecognition

This project uses the keras library to build a convolutional neural network based speech recognition model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

1) Make sure python 3 is installed on your device before going to next steps.
2) Project includes a #requirements.txt file. In terminal got 

### Installing

A step by step series of examples that tell you how to get a development env running
```
1) Clone the project onto your local device.
2) In terminal/cmd line, go to project directory and run # 'wandb setup'
```
3 a) To use the pretrained model use the following steps
```
1) Run # "wandb run python record.py"
2) Go to project directory, add that file to sub folder #Prediction in the folder #Predict
3) In terminal/cmd run # "rm .DS_Store" in both folders
4) Run predict.py
```
3 b) To train model on your data use the following steps
```
1) Collect recordings of the words required in .WAV format in length roughly 2 seconds.
2) Add recordings under labeled sub-folders in the data folder.
3) Run # "rm .DS_Store" in all subfolders
4) Run # "wandb run python audio.py"
5) Follow steps to use pretrained model to use the new model.
```

## Running the tests

Adding any of the following lines to main.py or audio.py (after the model is fit) can help with understanding the model.

1) 

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
