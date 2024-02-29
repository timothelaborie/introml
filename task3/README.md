# Food triplet classifier

## How I got first place

I started with this [Keras Siamese Network Example](https://keras.io/examples/vision/siamese_network/). In the middle of the page, you can see that their embeddings extractor is 

```python
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)
```

However, that one is not optimal, so I instead used the one from the DeepRanking paper: [DeepRanking GitHub](https://github.com/akarshzingade/image-similarity-deep-ranking/blob/master/deepRanking.py)

```python
x = Dense(4096, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.6)(x)
x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
```

I also tried tweaking it a bit but the original one performed the best.

For my feature extractor, I initially used MobileNet V2. This got me a score of 71%.

I then changed it to EfficientNet XL and got 74%.

I then realized the learning rate was way too high as it was overfitting after a single epoch. Changing it from the Adam default to 0.000001 improved it to 75%.

Next, I concatenated the features of XL and ResNet 50, and made a submission for every epoch. This got me 76.7%.

At this point the project deadline was over, but I felt like improving it further for fun.

I then tried various techniques like circle loss, or a different embeddings extractor, but the only thing that improved it was setting up a weighted ensemble of triplet loss models, each trained using their own feature extractor. In the end, I had 

```python
score_left = 0.5*row[0] + row[1] + row[2] + 0.5*row[3]
```

where the models are EfficientNet XL, ResNet50, CLIP, Vision Transformer, each trained on 7 epochs. Each "row" value is the model's predicted (left-right) distance.

This got me 77.7%

## Training code

https://github.com/timothelaborie/introml/blob/main/task3/main.ipynb

## Leaderboard Screenshot

![leaderboard](https://github.com/timothelaborie/introml/blob/main/task3/iml%20task%203.png)

There were about 400 teams.

First place with 77.7% accuracy (second place got 75.7% and there were about 400 teams)
