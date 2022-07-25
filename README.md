# OCR-Kanji

A hobby project. A light CNN which detects and classifies japanese kanjis from different difficulty levels (N5, N4 etc).
It contains a training data generator which creates japanese newspaper-like images with position and class labels.

## Training data generation
First have a look through the parameters in the below file and run it in order to generate (recommended) about 1000 training images and labels.
```Python
generate.py
```
You can look at the generated labels for an example image by running:
```Python
visualise.py
```

## Training the model
Model architecture is found in:
```Python
model.py
```
Run:
```Python
train.py
```
It utilises a data generator defined in:
```Python
dataset.py
```

## Predict
Run:
```Python
evaluate.py
```
The script takes the image defined in the file, splits it into patches to feed to the network and then it predicts the location and difficulty of the kanjis it found. It's not perfect.

## Improvements
Probably a good idea to add katakana and hiragana classes.
The training data generator should include larger characters, different fonts and diverse backgrounds.
