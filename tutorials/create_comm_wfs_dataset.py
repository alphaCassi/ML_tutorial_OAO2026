import sys

sys.path.insert(0, "/home/lucacor/PHD/OAO_school_Asiago_2026/AI4AO")

print(sys.path)




from mmengine import Config
import matplotlib.pyplot as plt
import torch


from AI4AO import PyramidWFS, ZernikeWFS, PhaseDataset, FramePreprocess, imshow, imshow_multiple, DeformableMirror


def symlog(x):
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))

def main(nModes = 20, size = 100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    paramfile = 'wfs_params_exp.py'

    AtmosParams = Config.fromfile(paramfile)['AtmosParams']
    WFSParams = Config.fromfile(paramfile)['WFSParams']
    LoopParams = Config.fromfile(paramfile)['LoopParams']
    DMParams = Config.fromfile(paramfile)['DMParams']

    AtmosParams['Scintillation'] =False
    AtmosParams["Nphases"] = 1

    dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)
    dataset.generateClosedLoop = True

    # create wfs mask and do the preprocessing
    wfs = PyramidWFS(WFSParams, device)
    wfs.BuildMask()
    wfs.BuildReferenceIntensity()

    framePreprocessor = FramePreprocess(WFSParams, wfs, device)
    framePreprocessor.ProcessReference(wfs.reference_intensity)

    # create the dm, with given IF anc actuators
    dm = DeformableMirror(WFSParams, DMParams, device)
    dm.offset_to_fit_number_of_actuators = 0.1
    dm.eval()

    # Create a M2C matrix using Zernike coeffs
    M2C = dm.MakeZernikeM2C(nModes=nModes)
    print(f"The M2C has shape {M2C.shape}")

    # create the "2d_pixel to modes" matrix (modes,H,W)
    modes = dm(M2C.T)

    # calculate the "modes to 2d_pixels" matrix (HxW, modes)
    z_inv = torch.linalg.pinv(dm(M2C.T).flatten(start_dim=-2))
    # print(f"The shape of z_inv is {z_inv.shape}")

    comm_tensor = torch.zeros((size, nModes))
    wfs_tensor = torch.zeros((size, 4, 46,46))

    # calculate a batch of data
    for i in range(size):
        batch = dataset[0]
        with torch.no_grad():

            ########### DM COMMANDS
            # from opd to zernike modes
            opd = batch["opd"] 
            # get the nmodes of this given opd
            proj_zern_modes = opd.squeeze(0).flatten() @ z_inv
            # proj_zern_modes = symlog(proj_zern_modes)

            # print(f"opd shape: {opd.shape}")


            ########### WFS IMAGES
            wfs_frame = wfs.Propagator(opd)
            preprocessed_frames = framePreprocessor.ProcessFrame(wfs_frame)

            # proj_zern_modes.append(comm_tensor)
            # preprocessed_frames.append(wfs_temspr)

            comm_tensor[i] = proj_zern_modes
            wfs_tensor[i] = preprocessed_frames

    return {"comm": torch.tensor(comm_tensor),
                "wfs": torch.tensor(wfs_tensor)}


if __name__ == "__main__":
    nModes = 50
    size = 5000
    data = main(nModes=nModes, size = size)
    torch.save(data, "dati_modello.pt")

    comm_ex = data["comm"][0]
    wfs_ex = data["wfs"][0]

    plt.figure(figsize = (15,8))
    plt.subplot(121)
    x = torch.arange(nModes)
    plt.plot(x, comm_ex)

    plt.subplot(122)
    # plt.imshow(wfs_ex[0])
    imshow(wfs_ex)

    plt.show()

