from GAN_generator import GAN_generator
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


folder = './' # location of parameter file
file = 'network_parameters.pt' # name of parameter file
checkpoint_parameter = torch.load(folder +  file, map_location=torch.device('cpu'))
noise_parameter = checkpoint_parameter['noise_parameter']
print(noise_parameter)
H_gen=[384,16384, 256, 128, 64, 1]
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
epoch = checkpoint_parameter['epoch']
real_label = 1
fake_label = 0
beta1 = 0.5
criterion = torch.nn.BCELoss()
lr = 0.0002


netG = GAN_generator(H_gen).float().to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
netG.load_state_dict(checkpoint_parameter['model_state_dict_gen'])

b_size = 25
D_in_gen = [b_size, 64, 6]


xplot=np.linspace(0,64*1.1,64)
yplot=np.linspace(1,16.36,64)
norm = Normalize(-35, 20)


f, axs = plt.subplots(5, 5, figsize=(12, 10))
file_number = 0
counter = 0
noise = (torch.randn(D_in_gen)).to(device)
output = netG(noise, None)
output = (output + 1) * (55 / 2) - 35
for i in range(0,5):
    for j in range(0,5):

        pcm = axs[i, j].pcolormesh(xplot, yplot, np.transpose(output.detach().numpy()[i + 5 * j][0]), norm=norm)
        title_str = 'Scene' + str(i)

        axs[i, j].tick_params(axis='both', which='major', labelsize='12')
        if i == 0:
            axs[j, i].set_ylabel('Generated \nAltitude [km]', fontsize=12)
        else:
            axs[j, i].tick_params(labelleft=False)
        if j == 4:
            axs[j, i].set_xlabel('Position [km]', fontsize=12)
        else:
            axs[j, i].tick_params(labelbottom=False)

f.tight_layout()
f.subplots_adjust(right=0.88)
cbar_ax = f.add_axes([0.9, 0.067, 0.025, 0.915])
cbar1= f.colorbar(pcm, cax=cbar_ax)
cbar1.set_label('Reflectivities [dBZ]', fontsize=12)
cbar1.ax.tick_params(labelsize=12)
print('image saved as: ', 'testepoch' + str(epoch) + '_GAN')
plt.savefig('testepoch' + str(epoch) + '_GAN')