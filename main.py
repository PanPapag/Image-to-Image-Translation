import os

DATASET_NAME = 'facades'
IMG_HEIGHT = 256
IMG_WIDTH = 256
SAMPLE_INTERVAL = 250
MODEL_INTERVAL = 10
EPOCHS = 200
LR = 0.0002
B1 = 0.5
B2 = 0.999


def main():
    os.makedirs("generated_images/{}".format(DATASET_NAME), exist_ok=True)
    os.makedirs("saved_models/{}".format(DATASET_NAME), exist_ok=True)

if __name__ == '__main__':
    main()