import torch
class GAN_discriminator (torch.nn.Module):
    def __init__(self, H):
        #for GAN
        # H=[5, 256, 128, 128, 5, 1, 64, 128, 256, 256, 4096, 1]
        #for CGAN
        # H =[8, 256, 128, 64, 8, 9, 64, 128, 256, 256, 4096, 1]
        super(GAN_discriminator, self).__init__()
        #region
        self.upsample0 = torch.nn.ConvTranspose2d(H[0],H[0],(4,1), stride=(4,1))
        self.convolution0 = torch.nn.Conv2d(H[0],H[1],(5,3),padding=(2,1))
        #relu
        self.batchNorm0 = torch.nn.BatchNorm2d(H[1])

        self.upsample1 = torch.nn.ConvTranspose2d(H[1], H[1], (4, 1), stride=(4, 1))
        self.convolution1 = torch.nn.Conv2d(H[1], H[2], (5, 3), padding=(2, 1))
        # relu
        self.batchNorm1 = torch.nn.BatchNorm2d(H[2])

        self.upsample2 = torch.nn.ConvTranspose2d(H[2], H[2], (2, 1), stride=(2, 1))
        self.convolution2 = torch.nn.Conv2d(H[2], H[3], (3, 3), padding=(1, 1))
        # relu
        self.batchNorm2 = torch.nn.BatchNorm2d(H[3])

        self.upsample3 = torch.nn.ConvTranspose2d(H[3], H[3], (2, 1), stride=(2, 1))
        self.convolution3 = torch.nn.Conv2d(H[3], H[4], (3, 3), padding=(1, 1))
        # relu
        self.batchNorm3 = torch.nn.BatchNorm2d(H[4])
        #endregion

        #concatenate

        self.convolution5 = torch.nn.Conv2d(H[5],H[6],(3,3),stride=(2,2),padding=(1,1))
        #relu

        self.convolution6 = torch.nn.Conv2d(H[6], H[7], (3, 3), stride=(2, 2),padding=(1,1))
        # relu

        self.convolution7 = torch.nn.Conv2d(H[7], H[8], (3, 3), stride=(2, 2),padding=(1,1))
        # relu

        self.convolution8 = torch.nn.Conv2d(H[8], H[9], (3, 3), stride=(2, 2),padding=(1,1))
        # relu

        #flatten
        self.dense9=torch.nn.Linear(H[10],H[11])
        self.sigmoid9 = torch.nn.Sigmoid()

    def forward(self, x, scene):
        #region
        if x != None:

            h_upsample0 = self.upsample0(x)
            h_conv0 = self.convolution0(h_upsample0)
            h_relu0 = torch.nn.functional.leaky_relu(h_conv0,0.2)
            h_batch0 = self.batchNorm0(h_relu0)

            h_upsample1 = self.upsample1(h_batch0)
            h_conv1 = self.convolution1(h_upsample1)
            h_relu1 = torch.nn.functional.leaky_relu(h_conv1, 0.2)
            h_batch1 = self.batchNorm1(h_relu1)

            h_upsample2 = self.upsample2(h_batch1)
            h_conv2 = self.convolution2(h_upsample2)
            h_relu2 = torch.nn.functional.leaky_relu(h_conv2, 0.2)
            h_batch2 = self.batchNorm2(h_relu2)

            h_upsample3 = self.upsample3(h_batch2)
            h_conv3 = self.convolution3(h_upsample3)
            h_relu3 = torch.nn.functional.leaky_relu(h_conv3, 0.2)
            h_batch3 = self.batchNorm3(h_relu3)
        #endregion
            h_conc4 = torch.cat((h_batch3, scene),1)
        else:
            h_conc4=scene
        h_conv5 = self.convolution5(h_conc4)
        h_relu5 = torch.nn.functional.leaky_relu(h_conv5, 0.2)

        h_conv6 = self.convolution6(h_relu5)
        h_relu6 = torch.nn.functional.leaky_relu(h_conv6, 0.2)

        h_conv7 = self.convolution7(h_relu6)
        h_relu7 = torch.nn.functional.leaky_relu(h_conv7, 0.2)

        h_conv8 = self.convolution8(h_relu7)
        h_relu8 = torch.nn.functional.leaky_relu(h_conv8, 0.2)

        h_flat9 = h_relu8.view(len(scene),-1)
        h_dense9 = self.dense9(h_flat9)
        h_out=self.sigmoid9(h_dense9)
        return h_out
