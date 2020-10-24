import h5py
from IceWaterPathMethod import IceWaterPathMethod
from GAN_generator import GAN_generator

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

folder = './' # location of parameter file
file= 'network_parameters_CGAN.pt' # name of parameter file
checkpoint_parameter = torch.load(folder + file, map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']


print(noise_parameter)
H_gen=[576,16384, 256, 128, 64, 1]
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
epoch = checkpoint_parameter['epoch']
real_label = 1
fake_label = 0
beta1 = 0.5
criterion = torch.nn.BCELoss()
lr = 0.0002

f_size = 14

netG = GAN_generator(H_gen).float().to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
netG.load_state_dict(checkpoint_parameter['model_state_dict_gen'])

b_size = 1
D_in_gen = [b_size, 1,1,64]



def t_count(t, val):
    elements_equal_to_value = torch.eq(t, val)
    as_ints = elements_equal_to_value.numpy()
    count = np.sum(as_ints)
    return count





location = './modis_cloudsat_data/'
file_string = location + 'CGAN_test_data_with_temp_conc_ver2.h5' #for CGAN 2015
#file_string = location + 'modis_cloudsat_ElevLatLong_test_data_conc_2016_ver2.h5' #for CGAN 2016
hf = h5py.File(file_string, 'r')

cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))

modis_scenes = torch.tensor(hf.get('modis_scenes'))
print('modis scenes input size: ', modis_scenes.shape)
temp_modis_scenes = torch.cat([modis_scenes[:, :, :, 0:3], modis_scenes[:, :, :, 4:9]], 3)
modis_scenes = temp_modis_scenes
print('modis after change size: ', modis_scenes.shape)
temperature = torch.tensor(hf.get('temperature'))
dataset = torch.utils.data.TensorDataset(cloudsat_scenes, modis_scenes,temperature)

print('number of missing values in modis ', t_count(modis_scenes,(float(-1))))



xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)


norm = Normalize(-35, 20)
f, axs = plt.subplots(8,5, figsize=(13,18))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                             num_workers=2)
file_number = 0
counter = 0
for i, data in enumerate(dataloader,0):
    if i > 4:
        break
    cs = data[0]

    print('cs first element: ',cs[0,0,0,0])
    string_ex = 'example' + str(i)
    modis = data[1]
    temperature_data = data[2]
    print(temperature_data.shape, 'shape of temperature data')

    modis = torch.transpose(modis, 1, 3)
    modis = torch.transpose(modis, 2, 3)
    print('modis shape', modis.shape)
    modis = modis.to(device)
    pl = axs[2,i].plot(xplot,np.transpose(np.array(modis[0,:,0,:])))
    axs[2,i].set_ylim([-1,1])
    axs[2,i].minorticks_on()
    axs[2, i].tick_params(axis='both', which='major', labelsize=f_size)
    axs[2, i].tick_params(axis='both', which='minor')
    axs[2,i].tick_params(labelbottom=False)
    if i == 0:
        axs[2, i].set_ylabel('Modis Input', fontsize=f_size)
    else:
        axs[2, i].tick_params(labelleft=False)

    cs = (cs + 1) * (55 / 2) - 35

    # Calculate cloud top height:
    altitudes_cs = np.zeros([1, 64])
    for k in range(0, 64):
        first_cloud_location_cs = 0
        for j in range(63, -1, -1):  # Loop backwards, starting from top of scene
            if cs[0,0, k, j] >= -15 and first_cloud_location_cs == 0:  # Set dBZ limit for cloud top detection
                altitudes_cs[0, k] = j  # save the index of the altitude where the cloud top is for each position
                first_cloud_location_cs = +1
    altitudes_cs = (altitudes_cs * 0.24) + 1  # Altitude of cloud top over sea level, [km]

    # Calculate indices for IWP
    indexes_zero = np.zeros([len(temperature_data), 1, 64, 1])

    for l in range(0, len(temperature_data)):
        for position in range(0, 64):
            for j in range(len(temperature_data[l, 0, position]) - 2, -1, -1):
                if (temperature_data[l, 0, position, j] - 273.15) * (temperature_data[l, 0, position, j + 1] - 273.15) <= 0:
                    indexes_zero[l, 0, position] = j + 1
                    break
                if j == 0:
                    if torch.max(temperature_data[l, 0, position, :]) <= 273.15:
                        indexes_zero[l, 0, position] = 63
                    else:
                        indexes_zero[l] = 0
                        print('All above zero: ', temperature_data[l, 0, position])

    indexes_zero_for_cs = np.ones([len(indexes_zero), 1, 64, 1]) * 63 - indexes_zero

    [IWP_cs_1, IWP_cs_2, IWP_cs_3] = IceWaterPathMethod(cs,indexes_zero_for_cs)
    cs = cs.to(device)
    pcm1 = axs[7, i].pcolormesh(xplot, yplot, np.transpose(cs.cpu().detach().numpy()[0,0]), norm=norm)
    title_str = 'Scene' + str(i)
    axs[7, i].minorticks_on()
    axs[7,i].set_xlabel('Position [km]', fontsize=f_size)
    if i == 0:
        axs[7, i].set_ylabel('Real \nAltitude [km]', fontsize=f_size)
    else:
        axs[7, i].tick_params(labelleft=False)
    axs[7, i].tick_params(axis='both', which='both', labelsize=f_size)

    color_array = ['blue','green','red','purple']
    for j in range(0,4):
        noise = (torch.randn(D_in_gen)).to(device)
        output = netG(noise, modis)
        output = (output + 1) * (55 / 2) - 35

        pcm = axs[7-(j+1), i].pcolormesh(xplot, yplot, output.detach().numpy()[0,0], norm=norm)
        title_str = 'Scene' + str(i)
        axs[7-(j+1), i].minorticks_on()
        axs[7-(j+1), i].tick_params(axis='both', which='major', labelsize=f_size)
        axs[7-(j+1), i].tick_params(axis='both', which='minor')
        axs[7-(j+1),i].tick_params(labelbottom=False)
        if i == 0:
            axs[7-(j+1), i].set_ylabel('Generated \nAltitude [km]', fontsize=f_size)
        else:
            axs[7 - (j + 1), i].tick_params(labelleft=False)

        with torch.no_grad():

            [IWP_generated_1, IWP_generated_2, IWP_generated_3] = IceWaterPathMethod(torch.transpose(output, 2, 3),indexes_zero_for_cs)
        pcm = axs[0, i].semilogy(xplot, (IWP_generated_1[0, 0]), color=color_array[j], label='IWP generated', basey=10, alpha=0.5)
        # Calculate cloud top height:
        altitudes_generated = np.zeros([1, 64])
        for k in range(0, 64):
            first_cloud_location_gen = 0
            for l in range(63, -1, -1):  # Loop backwards, starting from top of scene
                if output[0,0,l,k] >= -15 and first_cloud_location_gen == 0:  # Set dBZ limit for cloud top detection
                    altitudes_generated[0, k] = l  # save the index of the altitude where the cloud top is for each position
                    first_cloud_location_gen = +1
        altitudes_generated = (altitudes_generated * 0.24) + 1  # Altitude of cloud top over sea level, [km]

        alt = axs[1,i].plot(xplot, np.transpose(altitudes_generated), color = color_array[j], label ='CTH generated',alpha=0.3)
    alt_cs = axs[1,i].plot(xplot, np.transpose(altitudes_cs),color='black', label='CTH real')
    axs[1, i].minorticks_on()
    axs[1, i].tick_params(axis='both', which='major', labelsize=f_size)
    axs[1, i].tick_params(axis='both', which='minor')
    axs[1, i].tick_params(labelbottom=False)
    if i == 0:
        axs[1, i].set_ylabel('Cloud-Top \nHeight [km]', fontsize=f_size)
    else:
        axs[1, i].tick_params(labelleft=False)
    axs[1, i].set_ylim(1, 16.36)

    axs[0, i].minorticks_on()
    axs[0, i].tick_params(axis='both', which='major', labelsize=f_size)
    axs[0, i].tick_params(axis='both', which='minor')
    axs[0, i].set_ylim([10 ** (-2), 10 ** 1])

    pcm = axs[0, i].semilogy(xplot, (IWP_cs_1[0,0]), color='black', label='IWP real',basey=10)
    title_str = 'IWP generated' + str(i)
    axs[0, i].tick_params(labelbottom=False)
    if i == 0:
        axs[0, i].set_ylabel('IWP [g m⁻²]', fontsize=f_size)
    else:
        axs[0, i].tick_params(labelleft=False)
    #axs[0, i].set_aspect(1/4)

f.tight_layout()
f.subplots_adjust(right=0.88)
cbar_ax = f.add_axes([0.9, 0.0375, 0.025, 0.59]) #([distance from left, bottom height, width, height])
cbar1= f.colorbar(pcm1, cax=cbar_ax)
cbar1.set_label('Reflectivities [dBZ]', fontsize=f_size)
cbar1.ax.tick_params(labelsize=f_size)


print('image saved as: ', 'testepoch' + str(epoch) + '_CGAN')
plt.savefig('./early_epochs/cgan/testepoch' + str(epoch) + '_CGAN')

