import torch
import  numpy as np



def IceWaterPathMethod(cloudsat_scenes, zero_indices):
    scene_number = 0
    Z_m95 = np.power(10, cloudsat_scenes * 0.1)
    # Calculate IWC in g/m^3
    IWC = 0.149 * np.power(Z_m95, 0.681)
    IWC_ver2 = 0.198 * np.power(Z_m95, 0.701)
    IWC_ver3 = 0.108 * np.power(Z_m95, 0.770)
    IWC = IWC.reshape(-1,1,64,64)
    IWC_ver2 = IWC_ver2.reshape(-1,1,64,64)
    IWC_ver3 = IWC_ver3.reshape(-1,1,64,64)

    # Calculate altitude for freeze levelzero_index_at_position:
    IWC_above_freeze_1 = np.zeros([len(IWC),1,64,64])
    IWC_above_freeze_2 = np.zeros([len(IWC_ver2), 1, 64, 64])
    IWC_above_freeze_3 = np.zeros([len(IWC_ver3), 1, 64, 64])

    for scene in range(0,len(zero_indices)):
        for position in range(0,len(zero_indices[0,0])):
            zero_index_at_position = int(zero_indices[scene,0,position])
            IWC_above_freeze_1[scene,0,position,zero_index_at_position:] = IWC[scene,0,position,zero_index_at_position:]
            IWC_above_freeze_2[scene, 0, position,zero_index_at_position:] = IWC_ver2[scene, 0, position, zero_index_at_position:]
            IWC_above_freeze_3[scene, 0, position,zero_index_at_position:] = IWC_ver3[scene, 0, position, zero_index_at_position:]

    IWC_vertical_integrals = np.multiply(IWC_above_freeze_1, 240)
    IWC_vertical_integrals_ver2 = np.multiply(IWC_above_freeze_2, 240)
    IWC_vertical_integrals_ver3 = np.multiply(IWC_above_freeze_3, 240)

    IWC_vertical_integrals = torch.tensor(IWC_vertical_integrals)
    IWC_vertical_integrals_ver2 = torch.tensor(IWC_vertical_integrals_ver2)
    IWC_vertical_integrals_ver3 = torch.tensor(IWC_vertical_integrals_ver3)

    # Calculate IWP from IWC
    IWC_vertical_sum = torch.sum(IWC_vertical_integrals, dim=3)  # dim=1 for one scene, dim =3 if [x,x,x,x]
    IWC_vertical_sum_ver2 = torch.sum(IWC_vertical_integrals_ver2, dim=3)
    IWC_vertical_sum_ver3 = torch.sum(IWC_vertical_integrals_ver3, dim=3)
    # Convert from
    IWP_test_scene = IWC_vertical_sum * 1e-3
    IWP_test_scene_ver2 = IWC_vertical_sum_ver2 * 1e-3
    IWP_test_scene_ver3 = IWC_vertical_sum_ver3 * 1e-3

    return [IWP_test_scene, IWP_test_scene_ver2, IWP_test_scene_ver3]