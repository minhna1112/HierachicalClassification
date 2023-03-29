from torch.utils.data import Dataset
import torch
import os
from path import Path
import cv2
import torchvision

import  torchvision.transforms as transforms


class HierachicalDataset(Dataset):
    def __init__(self, root_path : str = "", genus_mapper : list =  []) -> None:
        """
         Initialize instance. This is the first step in initialization of the class. It sets the genus_mapper and fill_datadict methods
         
         Args:
         	 root_path: Path to root directory where data is stored
         	 genus_mapper: List of genus names to be mapped
         
         Returns: 
         	 A reference to the object to allow for method chaining ( optional ). Example :. from mne import
        """
        super().__init__()
        self.root_path = Path(root_path)
        
        self.genus_mapper = genus_mapper
        self.data_dict = {}
        self.file_names = []
        self.fill_datadict()

    def fill_datadict(self):
        """
         Fill data dictionary with data. This is called after loading the data and it will create the data dictionary
        """
        print("Filling data dictionary ... ")
        spec_idx = 0
        # For each genus in the genus_mapper and species in the genus_mapper iterate over all species in the genus_mapper and store the genus and species in self. data_dict.
        for (i, genus) in enumerate(self.genus_mapper):
            genus_name = genus["genus"]
            spec_names = genus["species"]
            # This function will create a dictionary of genus specie data for each file in the list of spec_names.
            for (j, spec_name) in enumerate(spec_names):
                # This function will create a dictionary of genus and specie data
                for file_name in os.listdir(self.root_path / genus_name / spec_name):
                    self.data_dict[file_name] = {
                        "genus" : {
                            "name": genus_name,
                            "idx": i
                        },
                        "specie": {
                            "name": spec_name,
                            "idx": spec_idx 
                        }
                    }

                spec_idx += 1
        self.file_names =  list(self.data_dict.keys())

        self.transforms = transforms.Compose([
            transforms.Resize([360, 640]),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Done ! ")
    
    def __getitem__(self, index :  int):
        """
         Returns the image at the given index. This is a function to be used by subclasses that need to access the data in order to get the image and genus and specie of a sample
         
         Args:
         	 index: Index of the sample to get
         
         Returns: 
         	 Tuple of the image in the format ( image_tensor genus_idx specie_idx )
        """
        query : str = self.file_names[index]
        sample = self.data_dict[query]
        genus_name = sample["genus"]["name"]
        genus_idx = sample["genus"]["idx"]
        
        spec_name = sample["specie"]["name"]
        specie_idx = sample["specie"]["idx"]
        
        image_tensor = torchvision.io.read_image(self.root_path / genus_name / spec_name / query)
        image_tensor = image_tensor.to(torch.float32) / 255.
        image_tensor = self.transforms(image_tensor)
        return image_tensor, torch.tensor(genus_idx), torch.tensor(specie_idx)
    
    def __len__(self):
        """
         Returns the number of files in the archive. This is useful for debugging the file system. If you want to know how many files are in the archive use : py : meth : ` FileArtist. __len__ ` instead.
         
         
         Returns: 
         	 The number of files in the archive or - 1 if there are no files in the archive or if the archive is an unreadable
        """
        return len(self.file_names)


# This function is used to generate a HierachicalTensor from the HierachicalDataset
if __name__=="__main__":
    genus_mapper = [
    {
        "genus" : "da",
        "species": ["00114", "00115", "00116", "00117", "00118"]
    },
    {
        "genus" : "gioi",
        "species": ["00094", "00095", "00096", "00097"]
    }
]
    dataset = HierachicalDataset("/mnt/d/minhna1112/data/hier_plants", genus_mapper=genus_mapper)
    image_Tensor, genus_idx, spec_idx = dataset[2]
    print(image_Tensor.dtype)