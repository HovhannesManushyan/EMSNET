from pathlib import Path

import pandas as pd
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms import Resize

CURRENT_PATH = Path(__file__).parent
DATA_PATH = CURRENT_PATH.parent.parent / 'data'
EVALUATION_PATH = DATA_PATH / 'Evaluation_Set'
TRAINING_PATH = DATA_PATH / 'Training_Set'
TEST_PATH = DATA_PATH / 'Test_Set'

# load the training labels csv file
TRAINING_LABELS = pd.read_csv(TRAINING_PATH / 'RFMiD_Training_Labels.csv')
TEST_LABELS = pd.read_csv(TEST_PATH / 'RFMiD_Testing_Labels.csv')
EVALUATION_LABELS = pd.read_csv(EVALUATION_PATH / 'RFMiD_Validation_Labels.csv')

# image paths for each set
TRAINING_IMAGES = [str(image) for image in (TRAINING_PATH / 'Training').glob('*.png')]
TEST_IMAGES = [str(image) for image in (TEST_PATH / 'Test').glob('*.png')]
EVALUATION_IMAGES = [str(image) for image in (EVALUATION_PATH / 'Evaluation').glob('*.png')]

class DataLoader:
    """Dataloader class to load the data and the labels from the training, test and evaluation sets
    """
    def __init__(
        self,
        image_size = (720,720),
        split = 'Train',
        augment = False
        ) -> None:
        self.resizer = Resize(image_size)
        self.split=split
        self.augment = augment

        if self.split=='Train':
            self.images = TRAINING_IMAGES
            self.labels = TRAINING_LABELS
        elif self.split=='Test':
            self.images = TEST_IMAGES
            self.labels = TEST_LABELS
        elif self.split=='Evaluation':
            self.images = EVALUATION_IMAGES
            self.labels = EVALUATION_LABELS
        else:
            raise ValueError('Split must be one of Train, Test or Evaluation')
        #TODO add padding,cropping and centering
        #create pytorch transforms augmentation pipeline consisting of rotation, flipping, and altering in brightness, saturation, contrast and hue
        if augment:
            self.augmenter = T.Compose([
                T.RandomRotation(20),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image_id = image_path.split('\\')[-1].split('.')[0]
        label = self.labels[self.labels['ID'] == int(image_id)]['Disease_Risk'].values[0]

        image = read_image(image_path)
        resized_image = self.resizer(image)

        if self.augment:
            resized_image = self.augmenter(resized_image)

        return resized_image, label
