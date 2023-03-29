import torch
from typing import List, Literal
from torchvision.models import resnet18

class HierachicalPredictionHead(torch.nn.Module):
    def __init__(self, in_features : int, 
        num_species: int, 
        num_genus: int, 
        num_species_each_genus: List[int],
        output_types : Literal["probs", "logits"] = "probs" ) -> None:
        
        """
         Initialize the model. This is called by __init__ to initialize the model. You can call it yourself as soon as you are done with the model but it will be more efficient to do it in one step.
         
         Args:
         	 in_features: The number of features in the data
         	 num_species: The number of species in the data
         	 num_genus: The number of genus in the data
         	 num_species_each_genus: The number of species each genus
         	 output_types: The type of output to be used
         
         Returns: 
         	 None or a Tensor of shape ( num_species num_genus ) that is the logits for each
        """
        super().__init__()
        
        self.species_head = torch.nn.Linear(in_features=in_features, out_features=num_species)
        
        assert len(num_species_each_genus) == num_genus
        self.num_genus = num_genus
        self.num_species_each_genus = num_species_each_genus
        self.output_types = output_types

    def forward(self, x):
        """
         Forward pass of the neural network. This is the first step in the forward pass of the neural network.
         
         Args:
         	 x: input data of shape [ batch_size num_genus ]
         
         Returns: 
         	 a tuple of ( logits species_probs ) where logits is the logits of the logit of the head of each
        """
        species_logits = self.species_head(x)

        species_probabilities= torch.nn.functional.softmax(species_logits, dim=1)
        species_probabilities = torch.split(species_probabilities, split_size_or_sections=self.num_species_each_genus, dim = 1)
        assert len(species_probabilities) == self.num_genus
        genus_probs =  [torch.sum(specie_prob, dim=1).unsqueeze(1) for specie_prob in species_probabilities]
        genus_probs = torch.cat(genus_probs, dim=1)
        genus_logits = torch.log(genus_probs)
        assert genus_logits.size(0) == species_logits.size(0)
        
        # Returns the logits and genus logits for the output types.
        if self.output_types in ["logits"]:
            return genus_logits, species_logits
        species_probabilities = torch.cat(species_probabilities, dim=1)
        return genus_probs, species_probabilities
        
class Resnet18FeatureExtractor(torch.nn.Module):
    def __init__(self) -> None:
        """
         Initialize ResNet18 pretrained model. This is called by PyTorch to initialize the model.
         
         
         Returns: 
         	 None ( just for compatibility with PyTorch ) or an instance of self ( which has been redefined
        """
        super().__init__()
        self.resnet18_fx = resnet18(pretrained=True)
        self.out_features = self.resnet18_fx.fc.in_features 
        self.resnet18_fx.fc = torch.nn.Identity()

    def forward(self, x):
        """
         Forward pass of ResNet - 18. This is the first step of the forward pass of the RNN.
         
         Args:
         	 x: input data of shape [ batch_size n_features ]
         
         Returns: 
         	 output of shape [ batch_size n_features ] with values in [ 0 1 ]. In case of error the error is raised
        """
        return self.resnet18_fx(x)

class HierachicalClassifier(torch.nn.Module):
    def __init__(self, genus_mapper : List[dict], output_types : str = "probs") -> None:
        """
         Initialize the model. This is the first step in the model initialization. It sets up the head and feature extractor to be used for prediction
         
         Args:
         	 genus_mapper: List of dictionaries mapping genus to features
         	 output_types: Type of output ( probabilistic or probabilities )
         
         Returns: 
         	 None Side Effects : Initializes the HierachicalPredictionHead object. Initializes the Resnet18FeatureExtractor
        """
        super().__init__()
        self.num_genus = len(genus_mapper)
        self.num_species_each_genus = [len(genus["species"]) for genus in genus_mapper]
        self.num_species = sum(self.num_species_each_genus)

        self.fx = Resnet18FeatureExtractor()
        self.head = HierachicalPredictionHead(
            in_features=self.fx.out_features,
            num_species=self.num_species,
            num_genus=self.num_genus,
            num_species_each_genus=self.num_species_each_genus,
            output_types  = output_types
        )

    def forward(self, x):
        """
         Forward pass of genetic algorithm. This is the entry point for the forward pass of the algorithm.
         
         Args:
         	 x: input data of shape ( nb_samples self. dim )
         
         Returns: 
         	 genus and species of shape ( nb_samples self. dim ) or ( n_samples None
        """
        x  = self.fx(x)
        genus, species = self.head(x)
        return  genus, species


# This is the main function to be called from the main module.
if __name__ == "__main__":
    image_tensor = torch.ones(8, 3, 320, 640)
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
    with torch.no_grad():
        model  = HierachicalClassifier(genus_mapper).eval()
        out = model(image_tensor)
        print([o.size() for o in out])