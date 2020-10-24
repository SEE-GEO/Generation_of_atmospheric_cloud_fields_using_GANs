from os import path
import h5py
from datetime import datetime
import torch

from GAN_generator import GAN_generator
from GAN_discriminator import GAN_discriminator
def Training_CGAN():

    H_gen=[576,16384, 256, 128, 64, 1]
    num_epochs=4000
    batch_size=64
    workers=2

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    print(device)

    # for CGAN
    H_disc =[8, 256, 128, 64, 8, 9, 64, 128, 256, 256, 4096, 1]

    netG = GAN_generator(H_gen).float().to(device)
    netD = GAN_discriminator(H_disc).float().to(device)

    real_label = 1
    fake_label = 0
    beta1=0.5
    criterion = torch.nn.BCELoss()
    lr=0.0002

    optimizerD= torch.optim.Adam(netD.parameters(),lr=lr, betas = (beta1,0.999))
    optimizerG= torch.optim.Adam(netG.parameters(),lr=lr, betas = (beta1,0.999))
    folder = '/'
    file = 'network_parameters_CGAN.pt'
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
        print('network parameters loaded')
    else:
        G_losses=[]
        D_losses=[]
        epoch_saved=-1
        noise_parameter = 0.7
        print('new network initialised')
    now = datetime.now().time()  # time object

    print("reading of files started: ", now)
    # code for reading nonconcatenated files
    location = './'
    file = 'modis_cloudsat_training_data_conc.h5'
    file_string = location + file
    hf = h5py.File(file_string, 'r')

    cloudsat_scenes = torch.tensor(hf.get('cloudsat_scenes'))

    modis_scenes = torch.tensor(hf.get('modis_scenes'))
    now = datetime.now().time()  # time object
    print(len(cloudsat_scenes)," files loaded: ", now)
    temp_modis_scenes = torch.cat([modis_scenes[:,:,:,0:3],modis_scenes[:,:,:,4:9]],3)
    modis_scenes=temp_modis_scenes
    dataset = torch.utils.data.TensorDataset(cloudsat_scenes,modis_scenes)

    for epoch in range(epoch_saved+1, num_epochs):

        now = datetime.now().time()  # time object
        print('epoch ', epoch, " started: ", now)
        # creating dataset from modis and cloudsat data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=workers)
        j=0
        for i, data in enumerate(dataloader,0):

            trainable =True
            #training discriminator with real data
            netD.zero_grad()
            real_cpu0 = data[0].to(device)
            b_size = real_cpu0.size(0)

            real_cpu0 = torch.transpose(real_cpu0,2,3)


            label=torch.full((b_size, ),real_label,device=device)

            D_in_disc = [b_size, 1, 64, 64]
            modis = data[1].to(device)
            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)

            modis = torch.transpose(modis,1,3)
            modis = torch.transpose(modis,2,3)

            output=netD(modis,real_cpu0+real_cpu1).view(-1)

            errD_real = criterion(output,label)

            errD_real.backward()
            D_x = output.mean().item()

            # training discriminator with generated data
            D_in_gen = [b_size, 1, 1, 64]
            noise = torch.randn(D_in_gen).to(device)

            fake = netG(noise,modis)
            label.fill_(fake_label)

            noise = noise_parameter * torch.randn(D_in_disc)
            real_cpu1 = noise.to(device)

            output = netD(modis,fake.detach() + real_cpu1).view(-1)

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

            output = netD(modis,fake + real_cpu1).view(-1)
            errG = criterion(output,label)

            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            j=j+b_size

                # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        noise_parameter = noise_parameter*0.8
        torch.save({
            'epoch': epoch,
            'model_state_dict_gen': netG.state_dict(),
            'optimizer_state_dict_gen': optimizerG.state_dict(),
            'loss_gen': G_losses,
            'model_state_dict_disc': netD.state_dict(),
            'optimizer_state_dict_disc': optimizerD.state_dict(),
            'loss_disc': D_losses,
            'noise_parameter' : noise_parameter
        }, 'network_parameters_CGAN.pt')
        if epoch%25 == 0:
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
            }, 'network_parameters_CGAN_' + ending)

        now = datetime.now().time()  # time object
        print('epoch ', epoch, " ended: ", now)

    print('done')
