from astropy.io import fits
import os
import yaml
import numpy as np

def load_rms_commands(root):
    path = os.path.join(root, "comm.fits")

    # get array with shape (timesteps, commands)
    time_commands = fits.getdata(path)

    # print(f"Time commands has shape {time_commands.shape}")

    # calculate the root mean square of the commands
    # over time in order to reduce the (timesteps, commands)
    # 2D array into a 1 dimensional vector with shape (commands)

    time_commands_squared = time_commands**2
    time_commands_squared_mean = np.mean(time_commands_squared, axis = 0)
    rms_commands = np.sqrt(time_commands_squared_mean)

    return rms_commands


def load_atm_params(root):
    with open(os.path.join(root, "params.yml")) as f:
        data = yaml.safe_load(f)

        seeing = np.array(data["seeing"]["constant"])
        L0 = np.array(data["atmo"]["L0"])
        v1 = np.array(data["wind_speed"]["constant"][0])
        v2 = np.array(data["wind_speed"]["constant"][1])
        v3 = np.array(data["wind_speed"]["constant"][2])
        v4 = np.array(data["wind_speed"]["constant"][3])
        

        return np.array([seeing, L0,v1, v2,v3,v4], dtype=np.float32)
    

def load_sr_int(root):
    path = os.path.join(root, "sr_int.fits")

    # get array with shape (timesteps, commands)
    sr_int = fits.getdata(path)

    return np.array([sr_int], dtype=np.float32)

def load_psfee_int(root):
    path = os.path.join(root, "psfee_int.fits")

    # get array with shape (timesteps, commands)
    psfee_int = fits.getdata(path)

    return np.array([psfee_int], dtype = np.float32)

def load_psffwhm_int(root):
    path = os.path.join(root, "psffwhm_int.fits")

    # get array with shape (timesteps, commands)
    psffwhm_int = fits.getdata(path)

    return np.array([psffwhm_int], dtype = np.float32)

