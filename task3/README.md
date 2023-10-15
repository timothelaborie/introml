# Food triplet classifier

## ML pipeline

- I extract visual features from the images using pre-trained models such as CLIP and Resnet
- For each visual model, I train an embedding MLP using triplet loss
- To classify triplets I compare embedding distances and ensemble the predictions of all the models

## Training code

https://github.com/timothelaborie/introml/blob/main/task3/main.ipynb

## Leaderboard position

First place with 77.7% accuracy (second place got 75.7%)
