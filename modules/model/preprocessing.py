from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np

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

class DataLoader(Dataset):
    """Dataloader class to load the data and the labels from the training, test and evaluation sets
    """
    def __init__(
        self,
        image_size = (720,720),
        split = 'Train',
        augment = False,
        upsample = None
        ) -> None:
        self.resizer = Resize(image_size,antialias = False)
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
        
        if upsample:
            #upsample is the ratio of the minority class to the majority class after resampling
            sampler = RandomOverSampler(sampling_strategy=upsample,random_state=42,)
            resampled_ids,resampled_labels = sampler.fit_resample(np.array(self.labels['ID']).reshape(-1,1),self.labels['Disease_Risk'])
            # replicate the self.images list to match the length of the resampled labels using the resampled_ids
            resampled_ids = np.array(resampled_ids).tolist()
            self.images = [self.images[id-1] for id in resampled_ids]
            # create a new dataframe with the resampled labels
            self.labels = pd.DataFrame({'ID':resampled_ids.reshape(-1),'Disease_Risk':resampled_labels})



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

        #change the resized image tensor to float32
        resized_image = resized_image.float()

        return resized_image, label


if __name__=='__main__':
    loader = DataLoader((64,64),split='Train',augment=True,upsample = 0.5)
    print(len(loader))
    print(loader[5])