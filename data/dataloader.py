from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch

def create_data_loader(dataset : Dataset)->DataLoader:
    """
     Creates a data loader that batches data. Note that this is different from the batch loader in that it does not shuffle the data before loading it.
     
     Args:
     	 dataset: The dataset to use. Must be a : py : class : ` pyspark. sql. DataFrame ` or : py : class : ` pyspark
    """
    batch_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=8,
        shuffle=True
    )
    return batch_loader

def create_dataloaders(dataset: Dataset, 

    split_ratio : float = 0.7,
    batch_size : int = 8,
    num_workers: int = 4):
    """
     Creates data loaders for training and validation. This is a helper function to create dataloaders for training and validation.
     
     Args:
     	 dataset: Dataset to use for training and validation. Must be of shape [ batch_size height width channels ]
     	 split_ratio: Percentage of images to split
     	 batch_size
     	 num_workers
    """
    
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Creating train set of {train_size} images and val set of {val_size} images")
    
    train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers = num_workers
            ) 

    val_loader =  DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers = num_workers
            )

    batch_loaders = {
        'train':
            train_loader,
        'val':
           val_loader
    }
    return batch_loaders

# This is a test case.
if __name__=="__main__":
    from dataset import HierachicalDataset 
    genus_mapper = [
        {
            "genus" : "da",
            "species": ['00114', '00115', '00116', '00117', '00118']
        },
        {
            "genus" : "gioi",
            "species": ['00094', '00095', '00096', '00097']
        }
    ]
    dataset = HierachicalDataset(root_path="/mnt/d/minhna1112/data/hier_plants", genus_mapper=genus_mapper)
    
    batch_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=8,
        shuffle=True
    )

    # Print the shape of the image tensor.
    for (image_tensor, genus_idx, specie_idx) in batch_loader:
        print(image_tensor.shape)
        print(genus_idx.shape)
        print(specie_idx.shape)