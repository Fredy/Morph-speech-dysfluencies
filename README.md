# Morph-speech-dysfluencies
Implementation of the paper: Automatic classification of speech dysfluencies in continuous speech based on similarity measures and morphological image processing tools

## Project setup

**You need python 3.6+**

1. Clone this repo and `cd` into its folder
2. If you have `pipenv` installed run `pipenv install` and then activate the venv with `pipenv shell`.  
   If you don' have `pipenv`:
   * Create a virtual environment `python -m venv env`
   * Activate the venv `env/source bin/activate`
   * Install the requirements `pip install -r requirements.txt`

## How to run it

1. First you need to download the UCLA dataset (release two). Specifically the _"for reading"_ dataset in .wav format. [Here](https://www.uclass.psychol.ucl.ac.uk/Release2/Reading/AudioOnly/wav/) you can download this dataset.
2. In the repo folder:
  * Run `python src/main.py <Path of an audio file>`
  * E.g.: `python src/main.py ../data/M_0053_10y2m_1.wav`

## Notes

* You can find some notes in `notes.md`.
* If you want to test the pipeline you can use the file `M_0053_10y2m_1.wav`, it contains a lot of recognizable repetitions and prolongations.
