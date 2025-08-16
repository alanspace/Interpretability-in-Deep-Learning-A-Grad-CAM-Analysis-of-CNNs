import torch 
from torch.utils.data import Dataset
import os  # Make sure os is imported
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def plot_heatmap(denorm_image, pred, heatmap):

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20,20), ncols=3)

    classes = ['cucumber', 'eggplant', 'mushroom']
    ps = torch.nn.Softmax(dim = 1)(pred).cpu().detach().numpy()
    ax1.imshow(denorm_image)

    ax2.barh(classes, ps[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    ax3.imshow(denorm_image)
    ax3.imshow(heatmap, cmap='magma', alpha=0.7)


class ImageDataset(Dataset):

    def __init__(self, df, data_dir = None, augs = None,):
        self.df = df
        self.augs = augs
        self.data_dir = data_dir 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the path and label from the dataframe
        row = self.df.iloc[idx]
        img_path = row.img_path
        label = row.label

        # Create the full, correct image path
        # e.g., 'GradCAM-Dataset' + '/' + 'train_images/cucumber/0.jpg'
        full_path = os.path.join(self.data_dir, img_path)

        # Read the image
        img = cv2.imread(full_path)

        # Check if the image was read successfully
        if img is None:
            print(f"ERROR: Could not read image at path: {full_path}")
            # Return empty tensors to avoid crashing the training loop
            return torch.tensor([]), torch.tensor(-1)

        # Convert from BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentations if they exist
        if self.augs:
            augmented = self.augs(image=img)
            img = augmented['image']

        # The label should also be a tensor
        label = torch.tensor(label, dtype=torch.long)

        return img, label
