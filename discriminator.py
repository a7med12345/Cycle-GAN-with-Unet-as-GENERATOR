import torch
from torch import nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(100,2)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        #return nn.AvgPool2d(x, x.size()[2:]).view(x.size()[0], -1)
        #x.view(x.size(0), -1) flatten tensor
        return self.fc(x.view(x.size(0), -1))

if __name__ == '__main__':
    net = Discriminator(3)
    print(net)

    test_x = Variable(torch.FloatTensor(1, 3, 100, 100))
    out_x = net(test_x)

    print(out_x.size())
