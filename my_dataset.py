
import os
import torch
from PIL import Image
from torchvision.transforms import transforms

# Image Pair Dataset
class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ds_type=1):
        # ds_type = 1 train-val dataset
        # ds_type = 0 test dataset
        self.data_dir = data_dir
        if ds_type:
            self.transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.RandomCrop(224), 
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor()
            ])
        
        # Build category tags: 'smoke' is 1, 'non-smoke' is 0
        self.labels = {'non-smoke': 0, 'smoke_early': 1, 'smoke_mid': 2, 'smoke_end': 3}
        # Traverse the data directory to build image pairs, paths, and labels
        self.image_pairs = []
        # "category" represent the folder name of the category
        for category in ['non-smoke', 'smoke_early', 'smoke_mid', 'smoke_end']:
            # "location" represent the folder name of the location
            for location in os.listdir(os.path.join(data_dir, category)):
                if ds_type == 1 and (location == 'pos_4' or location == 'pos_5'):
                    continue
                if ds_type == 0 and (location == 'pos_1_1' or location == 'pos_1_2' or location == 'pos_2_1' or location == 'pos_2_2' or location == 'pos_3'):
                    continue
                location_path = os.path.join(data_dir, category, location)
                # "image_folder" represent the folder name of the image pair
                for image_folder in os.listdir(location_path):
                    image_folder_path = os.path.join(location_path, image_folder)
                    visible_image_path = os.path.join(image_folder_path, 'image_vis_{}.png'.format(image_folder[4:]))
                    infrared_image_path = os.path.join(image_folder_path, 'image_ir_{}.png'.format(image_folder[4:]))
                    self.image_pairs.append({
                        'visible': visible_image_path,
                        'infrared': infrared_image_path,
                        'label': self.labels[category]
                    })

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        item = self.image_pairs[idx]
        # Open both type of images in RGB format
        visible_image = Image.open(item['visible']).convert('RGB')
        infrared_image = Image.open(item['infrared']).convert('RGB')
        label = item['label']

        # r, g, b = visible_image.split()

        if self.transform:
            visible_image = self.transform(visible_image)
            infrared_image = self.transform(infrared_image)

        return (visible_image, infrared_image), label

