from os import path
import h5py
from datetime import datetime
import torch
from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator


def Training_GAN():

    H_gen=[384,16384, 256, 128, 64, 1]
    batch_size=64
    workers=2
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print(device)

    #for GAN

    H_disc = [5, 256, 128, 128, 5, 1, 64, 128, 256, 256, 4096, 1]
    netG = GAN_generator(H_gen).to(device)
    netD = GAN_discriminator(H_disc).to(device)

    real_label = 1
    fake_label = 0
    beta1=0.5
    criterion = torch.nn.BCELoss()
    lr=0.0002

    optimizerD= torch.optim.Adam(netD.parameters(),lr=lr, betas = (beta1,0.999))
    optimizerG= torch.optim.Adam(netG.parameters(),lr=lr, betas = (beta1,0.999))

    folder = '//'
    file ='network_parameters.pt'
    if path.exists(folder + file):
        checkpoint = torch.load(folder + file)
        netG.load_state_dict(checkpoint['model_state_dict_gen'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict_gen'])
        netD.load_state_dict(checkpoint['model_state_dict_disc'])
        optimizerD.load_state_dict(checkpoint['optimizer_state_dict_disc'])
        epoch_saved = checkpoint['epoch']
        G_losses = checkpoint['loss_gen']
        D_losses = checkpoint['loss_disc']
        noise_parameter = checkpoint['noise_parameter']

    else:
        G_losses=[]
        D_losses=[]
        epoch_saved=-1
        noise_parameter = 0.7

    now = datetime.now().time()  # time object
    print("reading of cloudsat files started: ", now)
    for cloudsat_file in range(0,4999):
        location = '/cephyr/users/svcarl/Vera/cloud_gan/gan/temp_transfer/rr_data/training_data/'
        file_string = location + 'rr_data_2015_' + str(cloudsat_file).zfill(4) +'.h5'
        if path.exists(file_string):
            hf = h5py.File(file_string, 'r')

            cloudsat_scenes_temp = torch.tensor(hf.get('rr')).view(-1,1,64,64)
            if cloudsat_file == 0 :
                cloudsat_scenes=cloudsat_scenes_temp
            else:
                cloudsat_scenes = torch.cat([cloudsat_scenes,cloudsat_scenes_temp],0)
    now = datetime.now().time()  # time object
    print("reading of cloudsat files is done : ", now)
    for epoch in range(epoch_saved+1, 3000):
        if epoch%500 == 0:
            now = datetime.now().time()  # time object
            print("epoch number ", str(epoch),' started: ', now)

        dataloader = torch.utils.data.DataLoader(cloudsat_scenes, batch_size=batch_size, shuffle=True,
                                                 num_workers=workers)
        j=0
        for i, data in enumerate(dataloader,0):
            trainable =True
            #training discriminator with real data
            netD.zero_grad()
            real_cpu0 = data.to(device)
            b_size = real_cpu0.size(0)

            label=torch.full((b_size, ),real_label,device=device)

            #for GAN
            D_in_disc = [b_size, 1, 64, 64]

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)
            output=netD(None,real_cpu0+real_cpu1).view(-1)

            errD_real = criterion(output,label)

            errD_real.backward()
            D_x = output.mean().item()

            #training discriminator with generated data
            D_in_gen = [b_size, 64, 6]
            noise = torch.randn(D_in_gen).to(device)

            fake = netG(noise,None)
            label.fill_(fake_label)

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)

            output = netD(None,fake.detach() + real_cpu1).view(-1)

            errD_fake = criterion(output,label)

            D_G_z1 = output.mean().item()

            errD = errD_fake + errD_real
            if i%1!=0:
                trainable = False
            if trainable:
                errD_fake.backward()
                optimizerD.step()

            # update generator network
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)
            output = netD(None,fake + real_cpu1).view(-1)
            errG = criterion(output,label)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            j=j+b_size
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        noise_parameter = noise_parameter*0.9
        torch.save({
            'epoch': epoch,
            'model_state_dict_gen': netG.state_dict(),
            'optimizer_state_dict_gen': optimizerG.state_dict(),
            'loss_gen': G_losses,
            'model_state_dict_disc': netD.state_dict(),
            'optimizer_state_dict_disc': optimizerD.state_dict(),
            'loss_disc': D_losses,
            'noise_parameter' : noise_parameter
        }, 'network_parameters.pt')
        if epoch%200 == 0:
            ending = str(epoch) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict_gen': netG.state_dict(),
                'optimizer_state_dict_gen': optimizerG.state_dict(),
                'loss_gen': G_losses,
                'model_state_dict_disc': netD.state_dict(),
                'optimizer_state_dict_disc': optimizerD.state_dict(),
                'loss_disc': D_losses,
                'noise_parameter': noise_parameter
            }, 'network_parameters_' + ending)
        if epoch%500 == 0:
            now = datetime.now().time()  # time object
            print("epoch number ", str(epoch),' ended: ', now)
    print('done')