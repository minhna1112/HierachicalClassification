from trainers.dummy_trainer import DummyTrainer

from models.losses import HierachicalLoss
from models.resnet18 import HierachicalClassifier

from data.dataloader import create_dataloaders
from data.dataset import HierachicalDataset

import torch
import json
import math
import wandb

from tqdm import tqdm

class Trainer():
    def __init__(self,
        data_path: str,
        batch_size: int,
        split_ratio: float,
        output_types : str,
        freeze : bool  = True,
        w: float = 0.1, 
        workers : int = 8):
        """
         Initialize HierachicalClassifier. This is the method that should be called by subclasses to initialize the class.
         
         Args:
         	 batch_size: The batch size to use when training / eval
         	 split_ratio: The ratio of data to split into batches
         	 output_types: The type of output to use ( train eval )
         	 freeze: Freeze the model if True ( default )
         	 w: The weight of the model ( default 0. 1 )
         	 workers: The number of worker processes to use ( default 8
        """
       

        with open("genus_map.json", "r") as f:
            genus_mapper = json.loads(f.read())


        # data
        self.dataset = HierachicalDataset(data_path, genus_mapper) 
        # data loader
        self.batch_loaders = create_dataloaders(dataset=self.dataset, 
                                                split_ratio=split_ratio, 
                                                batch_size=batch_size,
                                                num_workers = workers)
        
        # if torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cpu. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cpu. is_available torch. cpu. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda. is_available torch. cuda
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.model = HierachicalClassifier(genus_mapper=genus_mapper)
        self.model = self.model.to(self.device) 
        self.losses =  HierachicalLoss(num_species= self.model.num_species, 
                                    num_genus=self.model.num_genus, 
                                    num_species_each_genus=self.model.num_species_each_genus,
                                    output_types=output_types, 
                                    w = w)
        self.batch_size = batch_size
        self.w = w
        self.freeze = freeze
        # Creates torch. Optimizer for the model.
        if self.freeze:
            self.optimizer = torch.optim.Adam(lr = 1e-5, params = self.model.head.parameters())
        else:
            self.optimizer = torch.optim.Adam(lr = 1e-5, params = self.model.parameters())
        
        metric_dict = {"acc" : 0., "num_correct" : 0, "accuracy_list": []}
        self.metric_dict = {
            "genus": metric_dict,
            "species": metric_dict
        }
        


    def train_step(self, inputs, targets):
        """
         Train the model one step. This is the entry point for the training process. You can call this directly if you want to do something like :.
         
         Args:
         	 inputs: The inputs to the model. It should be a list of numpy arrays.
         	 targets: The targets to the model. It should be a list of numpy arrays.
         
         Returns: 
         	 The loss of the model spec_loss : The loss of the model genus_loss : The
        """     
        
        self.model.train()
        self.losses.train()
        predictions = self.model(inputs)
        # print(predictions)
        total_loss, spec_loss, genus_loss = self.losses(predictions, targets)
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss, spec_loss, genus_loss

    def eval_step(self, inputs, targets):
        """
         Evaluate one step of evaluation. This is called by : meth : ` ~gensim. models. genus. GensimModel. evaluate `.
         
         Args:
         	 inputs: A tensor of shape ( batch_size self. num_anchors )
         	 targets: A tensor of shape ( batch_size self. num_an
        """
        with torch.no_grad():
            self.model.eval()
            self.losses.eval()
            genus_preds, spec_preds = self.model(inputs)
            genus_target, spec_target = targets
            # genus_target, spec_target = genus_target.to(torch.int64), spec_target.to(torch.int64)
            # Compute the softmax for the logits output types.
            if self.model.head.output_types in["logits"]:
                genus_preds = torch.nn.functional.softmax(genus_preds, dim=1)
                spec_preds = torch.nn.functional.softmax(spec_preds, dim=1)
            
            total_loss, spec_loss, genus_loss = self.losses((genus_preds, spec_preds ), targets)
            
            genus_preds = torch.argmax(genus_preds, dim=1)
            spec_preds = torch.argmax(spec_preds, dim=1)

            self.metric_dict["genus"]["num_correct"] += (genus_preds == genus_target).float().sum()
            self.metric_dict["species"]["num_correct"] += (spec_preds == spec_target).float().sum()

            # print(targets[1])
            # print(spec_preds)
            
            return total_loss, spec_loss, genus_loss

    def loop(self, num_epochs: int, loop_idx: int):
        """
         Run the plant in loop. This is the main loop for the plant. You can call it multiple times to run multiple loops.
         
         Args:
         	 num_epochs: Number of epochs to run the plant
         	 loop_idx: Index of the loop
        """
        wandb.init(
            project="plant-classify",
            config={
                    "freeze": self.freeze,
                    "w": self.w
                })
        self.config = wandb.config

        self.losses.species_bias = self.config["w"]
        # If freeze is set to true the optimizer will be used to freeze the model.
        if self.config["freeze"]:
            self.optimizer = torch.optim.Adam(lr = 1e-5, params = self.model.head.parameters())
        else:
            self.optimizer = torch.optim.Adam(lr = 1e-5, params = self.model.parameters())
        
        example_ct = 0
        # Train and evaluation of the dataset.
        for e in range(num_epochs):
            # Train and evaluation of the dataset.
            for phase in ["train", "val"]:
                n_steps_per_epoch = math.ceil(len(self.batch_loaders[phase].dataset) / self.batch_size)
                # Train and evaluation metrics for each batch.
                for batch_idx, (images, genus, species) in enumerate(tqdm(self.batch_loaders[phase])):
                    
                    example_ct += len(images)

                    images = images.to(self.device)
                    genus = genus.to(self.device)
                    species = species.to(self.device)
                    
                    # The total loss and spec loss for the current phase.
                    if phase in ["train"]:
                        total_loss, spec_loss, genus_loss = self.train_step(images, (genus, species))
                    # Evaluate the phase of the evaluation.
                    if phase in ["val"]:
                        total_loss, spec_loss, genus_loss = self.eval_step(images, (genus, species))
                    
                    
                    metrics = {f"{phase}/{phase}_loss": total_loss, 
                        f"{phase}/{phase}_genus_loss": genus_loss, 
                        f"{phase}/{phase}_species_loss": spec_loss, 
                        f"{phase}/epoch": (batch_idx + 1 + (n_steps_per_epoch * e)) / n_steps_per_epoch, 
                        f"{phase}/example_ct": example_ct}
                    
                    # Log metrics to wandb. log metrics to wandb
                    if batch_idx + 1 < n_steps_per_epoch:
                        # 🐝 Log train metrics to wandb 
                        wandb.log(metrics)
                
                # if phase is val then the metric_dict genus num_correct num_correct
                if phase in ["val"]:
                    genus_accuracy = (self.metric_dict["genus"]["num_correct"] / len(self.batch_loaders["val"].dataset))
                    spec_accuracy = (self.metric_dict["species"]["num_correct"] / len(self.batch_loaders["val"].dataset))
                    val_metrics = metrics.copy()
                    val_metrics["val/genus_acc"] = genus_accuracy
                    val_metrics["val/spec_acc"] = spec_accuracy
                    # print(val_metrics)
                    wandb.log(val_metrics)
        
        wandb.log({"w": self.w, **val_metrics})
        torch.save(self.model.state_dict(), f="checkpoints" + '/' + f'run_{loop_idx}.pt')
        wandb.finish()