
#from dataset import AOdataset
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import torch
from hydra.utils import instantiate
import os
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from utils.paths import get_output_dirs

def train_step(cfg, batch,
               optimizer,
               model,
               criterion,
               device

               ):
    '''
    This function calls the core steps of the train step
    '''

   
    # define input and output
    input, labels = batch

    input = input.to(device)
    labels = labels.to(device)

    # Zero your gradients for every batch (important!)
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(input)

    # Compute the loss and its gradients
    loss = criterion(outputs, labels)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss.item()


def val_step(cfg,batch, model, criterion, device):

    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

  
    

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    return loss.item()


def train(cfg,
          model,
          train_loader,
          val_loader,
          normalizer):
       
    # OPTIMIZER
    optimizer = instantiate(
            cfg.optimizer,
            params=model.parameters())
    
    # CRITERION
    criterion = instantiate(
        cfg.loss
    )


    output_dir = HydraConfig.get().runtime.output_dir

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ########### LISTS TO SAVE TRAIN AND VAL LOSS

    train_loss_list = []
    val_loss_list = []

    print(f"Start training!")


    device = torch.device(cfg.train.device)
    model.to(device)
    

    print(next(model.parameters()).device)
    
    ############ TRAIN LOOP
    for epoch in tqdm(range(cfg.train.epochs)):

        model.train()

        train_running_loss = 0
        train_num_samples = 0 
        
        for i, batch in enumerate(train_loader):
            # loop full single train step
            train_loss = train_step(cfg,batch,
                optimizer = optimizer,
                model = model,
                criterion = criterion, device = device)
            
            # get batch size for a correct handling of the average loss over one epoch
            inputs, labels = batch
            batch_size = inputs.size(0)

            train_running_loss += train_loss * batch_size
            train_num_samples += batch_size

        train_epoch_loss = train_running_loss / train_num_samples

        model.eval()

        # VAL LOSS
        val_running_loss = 0
        val_num_samples = 0
        for i, batch in enumerate(val_loader):

            val_loss = val_step(cfg,batch, model, criterion, device)

            # get batch size for a correct handling of the average loss over one epoch
            inputs, labels = batch
            batch_size = inputs.size(0)

            val_running_loss += val_loss * batch_size
            val_num_samples += batch_size

        val_epoch_loss = val_running_loss / val_num_samples


        print(f"Train Loss: {train_epoch_loss}, Val Loss: {val_epoch_loss}")

        ########## APPEND TRAIN AND VAL LOSSES
        train_loss_list.append(train_epoch_loss)
        val_loss_list.append(val_epoch_loss)

        # TODO
        if val_epoch_loss <= min(val_loss_list):

            best_val_loss= val_epoch_loss

            checkpoint_path = os.path.join(
            output_dir,
            "best_model.pt"
            )

            torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "criterion": criterion.state_dict()
                    if hasattr(criterion, "state_dict")
                    else None,
                "normalizer": normalizer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )
            
       

    # dirs = get_output_dirs()

    # torch.save(checkpoint, os.path.join(dirs["checkpoints"], "best_model.pt"))
    # torch.save(results, os.path.join(dirs["predictions"], "test_results.pt"))
            
    return checkpoint_path
                    

    ####### SAVE THE TRAIN AND VAL LOSSES
    # # TODO
    # torch.save(train_loss_list, "train_loss_list.pt")
    # torch.save(val_loss_list, "val_loss_list.pt")

    # # TODO SAVE PLOTS IN PDF E PNG










