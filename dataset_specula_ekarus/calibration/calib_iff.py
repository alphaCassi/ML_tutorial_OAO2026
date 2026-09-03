import specula
specula.init(-1)  # CPU
from specula.data_objects.iir_filter_data import IirFilterData

import os

# folder_name = "calib_iff"
# path = "/home/lucacor/PHD/SPIE/EKARUS/calibration"
# full_path = os.path.join(path, folder_name)
# os.makedirs(full_path, exist_ok=True)


# # Parametri base — adatta questi ai tuoi
# gain1 = [0.6,0.]   # un gain per gruppo di modi (come il tuo gain_ramp2)
# ff1   = [0.999, 0.999]  # forgetting factor per gruppo

# iir1 = IirFilterData.from_gain_and_ff(gain=gain1, ff=ff1)

# iir1.save(os.path.join(full_path,'iir1.fits'))
# print("Salvato iir1.fits")




# # Parametri base — adatta questi ai tuoi
# gain2 = [0.,0.4,0.2,0.1]   # un gain per gruppo di modi (come il tuo gain_ramp2)
# ff2   = [2,298,50,50]  # forgetting factor per gruppo

# iir2 = IirFilterData.from_gain_and_ff(gain=gain2, ff=ff2)

# iir1.save(os.path.join(full_path,'iir2.fits'))
# print("Salvato iir2.fits")



from specula.data_objects.iir_filter_data import IirFilterData

import specula
specula.init(-1)
from specula.data_objects.iir_filter_data import IirFilterData
import numpy as np

# iir1: 400 modi, gain solo sui primi 2
ff1  = [0.99]*400
gain1 = [0.5]*400
iir1 = IirFilterData.from_gain_and_ff(gain=gain1, ff=ff1)
iir1.save('calibration/iir1.fits')

# iir2: 400 modi, gain 0 sui primi 2, poi crescente per ordine
ff2   = [0.99]*2 + [0.99]*298 + [0.98]*50 + [0.97]*50
gain2 = [0.0 ]*2 + [0.5 ]*298 + [0.5 ]*50 + [0.5 ]*50
iir2 = IirFilterData.from_gain_and_ff(gain=gain2, ff=ff2)
iir2.save('calibration/iir2.fits')

# print("iir1 stabile:", iir1.is_stable())
# print("iir2 stabile:", iir2.is_stable())