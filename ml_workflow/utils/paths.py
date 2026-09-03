from hydra.core.hydra_config import HydraConfig
import os


def get_output_dirs():

    output_dir = HydraConfig.get().runtime.output_dir

    dirs = {
        "output": output_dir,
        "checkpoints": os.path.join(output_dir, "checkpoints"),
        "figures": os.path.join(output_dir, "figures"),
        "predictions": os.path.join(output_dir, "predictions"),
        "normalization": os.path.join(output_dir, "normalization"),
        "config": os.path.join(output_dir, "config"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs