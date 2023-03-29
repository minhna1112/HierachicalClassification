from torch.nn import CrossEntropyLoss, NLLLoss
import torch
from typing import List, Literal, Tuple

class HierachicalLoss(torch.nn.Module):
    def __init__(self, 
        num_species: int, 
        num_genus: int, 
        num_species_each_genus: List[int],
        output_types : Literal["probs", "logits"] = "probs",
        w : float = 0.1 ) -> None:
        """
         Initialize the class. This is called by __init__ to initialize the class. You can call it yourself as soon as you are done with the object.
         
         Args:
         	 num_species: The number of species in the model.
         	 num_genus: The number of genus in the model.
         	 num_species_each_genus: The number of species in each genus.
         	 output_types: The type of output to use.
         	 w: The weight to use for the bias. Default is 0. 1 which is the same as the L1 regularization parameter.
         
         Returns: 
         	 The instance of the class that was initialized with the parameters passed in. Note that the class does not have to be initialized beforehand
        """
        super().__init__()
        
        assert len(num_species_each_genus) == num_genus
        self.num_species = num_species
        self.num_genus = num_genus
        self.num_species_each_genus = num_species_each_genus
        self.output_types = output_types

        self.species_bias  = w

        # Set the species_loss genus_loss and species_loss for the output_types.
        if self.output_types in ["logits"]:
           self.species_loss = NLLLoss()
           self.genus_loss = NLLLoss()
        # This method sets the species_loss genus_loss and species_loss.
        if self.output_types in ["probs"]: 
            self.species_loss = CrossEntropyLoss()
            self.genus_loss = CrossEntropyLoss()

    def forward(self, predictions : Tuple[torch.Tensor], targets : Tuple[torch.Tensor]):
        """
         Forward function for model. This is called by : meth : ` ~gensim. models. Model. forward `
         
         Args:
         	 predictions: Predictions from each genus.
         	 targets: Predictions from each species. The order of the targets is important because of the batch size.
         
         Returns: 
         	 A tuple of ( predictions targets ) where predictions is a batch of predictions and targets is a batch of
        """
        genus_preds, spec_preds = predictions
        genus_target, spec_target = targets
        batch_size = genus_preds.size(0)
        # assert  genus_preds.size(0) == spec_preds.size(0)
        batch_size = genus_preds.size(0)

        # If the output_types in probs then the genus and spec targets are predicted.
        if self.output_types in ["probs"]:
            genus_target = torch.nn.functional.one_hot(genus_target, num_classes = self.num_genus)
            spec_target = torch.nn.functional.one_hot(spec_target, num_classes = self.num_species)
            genus_target = torch.tensor(genus_target, dtype=torch.float32)
            spec_target = torch.tensor(spec_target, dtype=torch.float32)

            # if self.train():
            #     assert genus_preds.size() == genus_target.size() 
            #     assert spec_preds.size() == spec_target.size() 

        w = self.species_bias 
        spec_loss = self.species_loss(spec_preds, spec_target)
        genus_loss =   self.genus_loss(genus_preds, genus_target)
        return w * spec_loss + (1.0 - w) * genus_loss, spec_loss, genus_loss 

