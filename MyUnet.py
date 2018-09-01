import torch
from torch.autograd import Variable


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel,output_channel,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x=self.relu(x)

        return x


class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self,x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor = factor, mode='bilinear')

    def forward(self,x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self,crop = True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self,x, tw, th):
        b,c,w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:,:,x1:x1 + tw, y1:y1 + th]

    def forward(self,x,y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x,h,w)
        return torch.cat((x,y),dim=1)


class UpBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,downsample = False):
        super(UpBlock, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.downsampling = DownSample()

    def forward(self,x):
        x1 = self.conv(x)
        if self.downsample:
            x = self.downsampling(x1)
        else:
            x = x1
        return x,x1

class DownBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,Upsample = False):
        super(DownBlock, self).__init__()
        self.Upsample = Upsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.upsampling = UpSample()
        self.crop = CropConcat()

    def forward(self,x,y):
        if self.Upsample:
            x = self.upsampling(x)
        x = self.crop(y,x)
        x = self.conv(x)
        return x


class Unet(torch.nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Unet, self).__init__()
        #Down Blocks
        self.conv_block1 = ConvBlock(input_channel,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)

        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64,output_channel,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self,x):

        x1 = self.conv_block1(x)
        x = self.downsample(x1)
        x2 = self.conv_block2(x)
        x= self.downsample(x2)
        x3 = self.conv_block3(x)
        x= self.downsample(x3)
        x4 = self.conv_block4(x)
        x = self.downsample(x4)
        x5 = self.conv_block5(x)

        x = self.upsample(x5)
        x = self.crop(x4, x)
        x = self.conv_block6(x)

        x = self.upsample(x)
        x = self.crop(x3,x)
        x = self.conv_block7(x)

        x= self.upsample(x)
        x= self.crop(x2,x)
        x = self.conv_block8(x)

        x = self.upsample(x)
        x = self.crop(x1,x)
        x = self.conv_block9(x)


        x = self.last_conv(x)

        return x

class Unet2(torch.nn.Module):
    def __init__(self,input_channel,output_channel,change_size=False):
        super(Unet2, self).__init__()
        #Down Blocks
        self.upblock1 = UpBlock(input_channel,64,downsample= change_size)
        self.upblock2 = UpBlock(64,128,downsample= change_size)
        self.upblock3 = UpBlock(128,256,downsample= change_size)
        self.upblock4 = UpBlock(256,512,downsample= change_size)
        self.upblock5 = UpBlock(512,1024,downsample= change_size)

        #Up Blocks
        self.downblock1 = DownBlock(1024+512, 512,Upsample= change_size)
        self.downblock2 = DownBlock(512+256, 256,Upsample= change_size)
        self.downblock3 = DownBlock(256+128, 128,Upsample= change_size)
        self.downblock4 = DownBlock(128+64, 64,Upsample= change_size)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64,output_channel,1)



        #self.downsample = DownSample()
        #self.upsample =   UpSample()

    def forward(self,x):

        x,x1 = self.upblock1(x)
        x,x2 = self.upblock2(x)
        x,x3 = self.upblock3(x)
        x,x4 = self.upblock4(x)
        x,x5 = self.upblock5(x)

        x = self.downblock1(x5,x4)
        x = self.downblock2(x,x3)
        x = self.downblock3(x,x2)
        x = self.downblock4(x,x1)

        x = self.last_conv(x)

        return x

if __name__ == '__main__':
    net = Unet2(3,3)
    print(net)

    test_x = Variable(torch.FloatTensor(1, 3, 100, 100))
    out_x = net(test_x)

    print(out_x.size())