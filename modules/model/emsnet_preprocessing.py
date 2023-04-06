from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from tifffile import imread
import torch
import torchvision.transforms as T
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize, ToTensor
from PIL import Image

CURRENT_PATH = Path(__file__).parent
DATA_PATH = CURRENT_PATH.parent.parent / 'data'
EVALUATION_PATH = DATA_PATH / 'Evaluation_Set'
TRAINING_PATH = DATA_PATH / 'Training_Set'
TEST_PATH = DATA_PATH / 'Test_Set'
ODIR_PATH = DATA_PATH / 'preprocessed_images'
MURED_PATH = DATA_PATH / 'Mured'

# load the training labels csv file
try:
    TRAINING_LABELS = pd.read_csv(TRAINING_PATH / 'RFMiD_Training_Labels.csv')
    TEST_LABELS = pd.read_csv(TEST_PATH / 'RFMiD_Testing_Labels.csv')
    EVALUATION_LABELS = pd.read_csv(EVALUATION_PATH / 'RFMiD_Validation_Labels.csv')
except:
    raise RuntimeError('Missing RFMiD data')
# image paths for each set
TRAINING_IMAGES = [str(image) for image in (TRAINING_PATH / 'Training').glob('*.png')]
TEST_IMAGES = [str(image) for image in (TEST_PATH / 'Test').glob('*.png')]
EVALUATION_IMAGES = [str(image) for image in (EVALUATION_PATH / 'Evaluation').glob('*.png')]

# write a function which finds the extension of a file given a path to that file which is missing the extension
def find_extension(path):
    available_extensions = ['.png', '.jpg', '.jpeg','.tif']
    for extension in available_extensions:
        if Path(path + extension).exists():
            return path + extension
    raise ValueError(f'image at {path} does not have a valid extension')

def remove_borders(tensor):
    """
    Removes black edges from a PyTorch tensor.

    Args:
        tensor (torch.Tensor): A PyTorch tensor of shape (channel, height, width).

    Returns:
        torch.Tensor: A tensor with black edges removed.
    """
    # Compute the sum of the pixel values in each row and column
    row_sums = torch.sum(tensor, dim=(0, 2))
    col_sums = torch.sum(tensor, dim=(0, 1))

    # Find the first and last non-zero row and column indices
    first_row = torch.nonzero(row_sums).min().item()
    last_row = torch.nonzero(row_sums).max().item()
    first_col = torch.nonzero(col_sums).min().item()
    last_col = torch.nonzero(col_sums).max().item()

    # Create a new tensor with the new size
    new_tensor = tensor[:, first_row:last_row+1, first_col:last_col+1]

    return new_tensor
    

class EMSNETDataLoader(Dataset):
    """Dataloader class to load the data and the labels from the training, test and evaluation sets
    """
    def __init__(
        self,
        image_size = (720,720),
        split = 'Train',
        augment = False,
        upsample = None,
        edge_removal = False,
        load_odir = False, #only used for GAN pretraining. Loads only the ODIR dataset
        load_mured = False
        ) -> None:
        self.resizer = Resize(image_size,antialias = False)
        self.split=split
        self.augment = augment
        self.edge_removal = edge_removal
        self.load_odir = load_odir
        self.load_mured = load_mured

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

        if self.load_odir:
            df = pd.read_csv(DATA_PATH / 'full_df.csv')
            df['Disease_Risk'] = df.apply(
                lambda x: 1 if (x['Left-Diagnostic Keywords']!='normal fundus' or x['Right-Diagnostic Keywords']!='normal fundus') else 0, axis = 1
                )
            df_lefts = df[['Left-Fundus', 'Disease_Risk']]
            df_lefts.columns = ['image', 'Disease_Risk']
            df_rights = df[['Right-Fundus', 'Disease_Risk']]
            df_rights.columns = ['image', 'Disease_Risk']
            total_df = pd.concat([df_lefts, df_rights])
            total_df['image'] = total_df['image'].apply(lambda x: str(ODIR_PATH / x))
            total_df = pd.DataFrame(
                {'ID':list(range(1,len(total_df)+1)),
                'Disease_Risk':total_df['Disease_Risk'],
                'image':total_df['image']}
                )
            existing_images = total_df['image'].apply(lambda x: Path(x).exists())
            total_with_existing_images = total_df[existing_images]
            self.labels = total_with_existing_images
            self.images = total_with_existing_images['image'].tolist()

        if self.load_mured:
            df_train = pd.read_csv(MURED_PATH / 'train_data.csv')
            df_eval = pd.read_csv(MURED_PATH / 'val_data.csv')
            total_df = pd.concat([df_train, df_eval])
            total_df['image'] = total_df['ID'].apply(lambda x: str(MURED_PATH / 'images' / x))
            total_df['image'] = total_df['image'].apply(lambda x: find_extension(x))
            total_df['Disease_Risk'] = total_df['NORMAL'].apply(lambda x: 0 if x==1 else 1)
            total_df = pd.DataFrame(
                {'ID':list(range(1,len(total_df)+1)),
                'Disease_Risk':total_df['Disease_Risk'],
                'image':total_df['image']}
                )
            self.df = total_df
            existing_images = total_df['image'].apply(lambda x: Path(x).exists())
            total_with_existing_images = total_df[existing_images]
            self.labels = total_with_existing_images
            self.images = total_with_existing_images['image'].tolist()

        if upsample:
            #TODO fix upsampling
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
            if augment=='classic':
                self.augmenter = T.Compose([
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(p = 0.1),
                    T.RandomVerticalFlip(p = 0.1),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                ])
            elif augment=='winner':
                self.augmenter = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3,brightness_limit = (-0.2,0.2),contrast_limit = (-0.2,0.2)),
                    A.MedianBlur(p=0.3,always_apply=False,blur_limit=5),
                    A.IAAAdditiveGaussianNoise(p=0.5,scale=(0,0.15*255)),
                    A.HueSaturationValue(hue_shift_limit=10,sat_shift_limit=10,val_shift_limit=10,p=0.3),
                    A.Cutout(p=0.5,max_h_size=20,max_w_size=20,num_holes=5)
                ])
            else:
                raise ValueError('Senc ban chka ara')


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        # permute = [2, 1, 0]
        image_path = self.images[index]
        image_id = image_path.split('\\')[-1].split('.')[0]
        if (not self.load_odir) and (not self.load_mured):
            label = self.labels[self.labels['ID'] == int(image_id)]['Disease_Risk'].values[0]
        elif self.load_odir or self.load_mured:
            try:
                label = self.labels[self.labels['image'].str.contains(image_id)]['Disease_Risk'].values[0]
            except IndexError:
                label = 0
                print(f"Didn't find a label for image in index {image_id}. Defaulting to 0")

        if not image_path.endswith('.tif'):
            image = read_image(image_path)
        else:
            image = Image.open(image_path)
            # image = imread(image_path)
            image = ToTensor()(image)
            image*=255
        # image = image[permute]
        #convert the white background to black and remove the black edges of the image
        if self.edge_removal:
            image = remove_borders(image)
            # image = image[:,image.sum(axis=0)!=0]
            # image = image[image.sum(axis=1)!=0,:]

        resized_image = self.resizer(image)

        if self.augment:
            if self.augment == 'classic':
                resized_image = self.augmenter(resized_image)
            elif self.augment == 'winner':
                array_image = resized_image.permute(1,2,0).numpy()
                augmented_image = self.augmenter(image=array_image)
                resized_image = torch.from_numpy(augmented_image['image']).permute(2,0,1)
            else:
                raise ValueError('Senc ban chka ara')
        #change the resized image tensor to float32
        resized_image = resized_image.float()

        return resized_image, label


if __name__=='__main__':
    loader = DataLoader((64,64),split='Train',augment=True,edge_removal=True)
    print(len(loader))
    print(loader[5])
