import os
import numpy as np
import torch
from plots.plot import plot_output

from hydra.core.hydra_config import HydraConfig


def test_step(
    batch,
    model,
    criterion,
    device,
):

    inputs, labels = batch

    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():

        outputs = model(inputs)

        loss = criterion(outputs, labels)

    return loss.item(), outputs, labels


def test(
    cfg,
    model,
    criterion,
    test_loader,
    normalizer,
):

    print("Start testing!")

    ##########################################################
    # DEVICE
    ##########################################################

    device = torch.device(cfg.train.device)

    model.to(device)
    model.eval()

    ##########################################################
    # OUTPUT DIRECTORY
    ##########################################################

    output_dir = HydraConfig.get().runtime.output_dir

    ##########################################################
    # STORAGE
    ##########################################################

    predictions = []
    targets = []
    losses = []

    ##########################################################
    # LOOP
    ##########################################################

    for batch in test_loader:

        loss, outputs, labels = test_step(
            batch=batch,
            model=model,
            criterion=criterion,
            device=device,
        )

        losses.append(loss)

        predictions.append(outputs.cpu())

        targets.append(labels.cpu())

    ##########################################################
    # CONCATENATE
    ##########################################################

    predictions = torch.cat(predictions, dim=0)

    targets = torch.cat(targets, dim=0)

    ##########################################################
    # DENORMALIZE
    ##########################################################

    predictions = normalizer.inverse_output(predictions)

    targets = normalizer.inverse_output(targets)

    ##########################################################
    # SAVE
    ##########################################################

    results = {
        "predictions": predictions,
        "targets": targets,
        "test_loss": float(np.mean(losses)),
    }

    torch.save(
        results,
        os.path.join(output_dir, "test_results.pt"),
    )

   

    print(f"Test loss: {results['test_loss']:.6f}")
    print(f"Results saved in {output_dir}/test_results.pt")
    

    ############# PLOT

    plot_output(predictions, targets, output_dir)

    print(f"Plots saved!")

    return results