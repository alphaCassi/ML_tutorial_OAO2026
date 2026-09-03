import matplotlib.pyplot as plt
import os

def plot_output(predictions, targets, output_dir):
    labels = ["Seeing", "L0", "v1", "v2", "v3", "v4"]
    fig, ax = plt.subplots(1, len(labels), figsize=(12, 5))

    

    for i in range(len(labels)):

        ax[i].scatter(targets[:, i], predictions[:, i], s=10)

        xmin = min(targets[:, i].min(), predictions[:, i].min())
        xmax = max(targets[:, i].max(), predictions[:, i].max())

        ax[i].plot([xmin, xmax], [xmin, xmax], "r--")

        ax[i].set_xlabel(f"True {labels[i]}")
        ax[i].set_ylabel(f"Predicted {labels[i]}")
        ax[i].set_title(labels[i])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "true_vs_pred.png"), dpi=300)
    plt.close(fig)
