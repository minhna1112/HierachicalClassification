import torch
import torchvision
import  torchvision.transforms as transforms
import json

from models.resnet18 import HierachicalClassifier 

from PIL import Image
from typing import Tuple

def tensor2label(preds : torch.Tensor):
    preds = torch.max(preds, dim=-1)
    label = int(preds.indices)
    prob = float(preds.values)
    return label, prob

class Infer():
    def __init__(self, 
                 weight_path : str = "",

                 ) -> None:
        """
         Initialize the genus_map class. This is called by __init__ and should not be called directly
        """
        self.transforms = transforms.Compose([
            transforms.Resize([360, 640]),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open("genus_map.json", "r") as f:
            genus_mapper = json.loads(f.read())

        self.net = HierachicalClassifier(genus_mapper)
        self.net.load_state_dict(torch.load(weight_path))
        self.genuses = [genus["genus"] for (i, genus) in enumerate(genus_mapper)]
        self.species = [specie for specie in [[specie for specie in genus["species"]] for genus in genus_mapper]]    

    def run(self, image_path: str):
        """
         Loads and transforms an image. This is the main entry point for the pipeline. It takes a path to an image and transforms it to a 3 - dimentional tensor of shape ( height width 3 )
         
         Args:
         	 image_path: The path to the
        """
        image_tensor = torchvision.io.read_image(image_path)
        image_tensor = image_tensor.to(torch.float32) / 255.
        image_tensor = self.transforms(image_tensor)
        image_tensor = image_tensor.unsqueeze(0) 
        with torch.no_grad():
            self.net.eval()
            genus_preds, spec_preds = self.net(image_tensor)
            genus_preds = tensor2label(genus_preds)
            spec_preds = tensor2label(spec_preds)

            print(f"Predicted Genus: {genus_preds[0]} of {genus_preds[1]*100} %") 
            print(f"Predicted Specie: {spec_preds[0]} of {spec_preds[1]*100} %")            

if __name__=="__main__":

    inferer = Infer(weight_path="checkpoints/run_8.pt")
    inferer.run(image_path="/home/minhna4lab/Downloads/ST_00114_000218.jpg")

        




