import torch


class GAN_generator (torch.nn.Module):

    def __init__(self, H):
        #for gan
        # H=[384,16384, 256, 128, 64, 1]
        #for cgan
        # H=[576,16384, 256, 128, 64, 1]
        super(GAN_generator, self).__init__()

        self.dense0 = torch.nn.Linear(H[0],H[1])
        #Relu
        self.batchNorm0 = torch.nn.BatchNorm2d(H[2])

        self.upsample1 = torch.nn.ConvTranspose2d(H[2],H[2],2, stride=2)
        self.convolution1 = torch.nn.Conv2d(H[2],H[2],3, stride=1, padding=1)
        # Relu
        self.batchNorm1 = torch.nn.BatchNorm2d(H[2])

        self.upsample2 = torch.nn.ConvTranspose2d(H[2],H[2],2,stride=2)
        self.convolution2 = torch.nn.Conv2d(H[2],H[3],3,stride=1,padding=1)
        # Relu
        self.batchNorm2 = torch.nn.BatchNorm2d(H[3])

        self.upsample3 = torch.nn.ConvTranspose2d(H[3], H[3],2,stride=2)
        self.convolution3 = torch.nn.Conv2d(H[3], H[4],3,stride=1,padding=1)
        # Relu
        self.batchNorm3 = torch.nn.BatchNorm2d(H[4])

        self.convolution4 = torch.nn.Conv2d(H[4],H[5],3,stride=1,padding=1)
        self.tanh4 = torch.nn.Tanh()

    def forward(self,x, modis):
        if modis is None:
            h_conc0 = x.view(len(x),-1)
        else:
            h_conc0 = torch.cat([x.reshape(len(x),-1),modis.reshape(len(modis),-1)],dim=1)
        h_relu0 = self.dense0(h_conc0).clamp(min=0)
        h_relu0 = h_relu0.view(len(x),256,8, 8)
        h_norm0 = self.batchNorm0(h_relu0)
        h_upsample1 = self.upsample1(h_norm0)
        h_relu1 = self.convolution1(h_upsample1).clamp(min=0)
        h_norm1 = self.batchNorm1(h_relu1)
        h_upsample2 = self.upsample2(h_norm1)
        h_relu2 = self.convolution2(h_upsample2).clamp(min=0)
        h_norm2 = self.batchNorm2(h_relu2)
        h_upsample3 = self.upsample3(h_norm2)
        h_relu3 = self.convolution3(h_upsample3).clamp(min=0)
        h_norm3 = self.batchNorm3(h_relu3)
        h_convolution4 = self.convolution4(h_norm3)
        out = self.tanh4(h_convolution4)
        return out