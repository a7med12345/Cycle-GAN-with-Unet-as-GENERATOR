from torch import nn
from torch.autograd import Variable
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torch.optim as optimizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyUnet import Unet2
from discriminator import Discriminator
from Dataset import ImageDataset
import argparse



def train_one_step( use_cuda,
                    netG_A2B,
                    netG_B2A,
                    netD_A,
                    netD_B,
                    real_A,
                    real_B,
                    optimizers,
                    iteration,writer):


    batch_size = real_A.size()[0]

    Tensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

    G_criterion = nn.L1Loss()
    D_criterion = nn.CrossEntropyLoss()


    if use_cuda:
        G_criterion = G_criterion.cuda()
        D_criterion = D_criterion.cuda()

    ##Optimizers
    optimizer_G1 = optimizers['G1']
    optimizer_G2 = optimizers['G2']
    optimizer_D1 = optimizers['D1']
    optimizer_D2 = optimizers['D2']



    ###### Generators A2B and B2A ######
    optimizer_G1.zero_grad()
    optimizer_G2.zero_grad()

    # Identity loss

    # G_A2B(B) should equal B if real B is fed
    same_B = netG_A2B(real_B)
    loss_identity_B = G_criterion(same_B, real_B) * 5.0
    # G_B2A(A) should equal A if real A is fed
    same_A = netG_B2A(real_A)
    loss_identity_A = G_criterion(same_A, real_A) * 5.0

    # GAN loss

    fake_B = netG_A2B(real_A)
    pred_fake = netD_B(fake_B)
    loss_GAN_A2B = D_criterion(pred_fake, target_real)

    fake_A = netG_B2A(real_B)
    pred_fake = netD_A(fake_A)
    loss_GAN_B2A = D_criterion(pred_fake, target_real)

    # Cycle loss

    recovered_A = netG_B2A(fake_B)
    loss_cycle_ABA = G_criterion(recovered_A, real_A) * 10.0

    recovered_B = netG_A2B(fake_A)
    loss_cycle_BAB = G_criterion(recovered_B, real_B) * 10.0

    # Total loss
    loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB


    loss_G.backward()

    optimizer_G1.step()
    optimizer_G2.step()

    ###################################

    ###### Discriminator A ######

    optimizer_D1.zero_grad()

    # Real loss
    pred_real = netD_A(real_A)
    loss_D_real = D_criterion(pred_real, target_real)

    # Fake loss
    #fake_A = fake_A_buffer.push_and_pop(fake_A)
    pred_fake = netD_A(fake_A.detach())
    loss_D_fake = D_criterion(pred_fake, target_fake)

    # Total loss
    loss_D_A = (loss_D_real + loss_D_fake) * 0.5
    loss_D_A.backward()


    optimizer_D1.step()
    ###################################

    ###### Discriminator B ######

    optimizer_D2.zero_grad()

    # Real loss
    pred_real = netD_B(real_B)
    loss_D_real = D_criterion(pred_real, target_real)

    # Fake loss
    #fake_B = fake_B_buffer.push_and_pop(fake_B)
    pred_fake = netD_B(fake_B.detach())
    loss_D_fake = D_criterion(pred_fake, target_fake)

    # Total loss
    loss_D_B = (loss_D_real + loss_D_fake) * 0.5
    loss_D_B.backward()

    optimizer_D2.step()
    ###################################

    ##write to tensorboard
    if iteration%400==0:
        #write images
        writer.add_image('Training/Generated/A_B', make_grid(fake_B, nrow=8, normalize=True), iteration)
        writer.add_image('Training/Generated/B_A', make_grid(fake_A, nrow=8, normalize=True), iteration)
        writer.add_image('Training/Input/B', make_grid(real_B, nrow=8, normalize=True), iteration)
        writer.add_image('Training/Input/A', make_grid(real_A, nrow=8, normalize=True), iteration)

        #write losses
        writer.add_scalar('Training/DiscriminatorA/loss', loss_D_A, iteration)
        writer.add_scalar('Training/DiscriminatorB/loss', loss_D_B, iteration)

        writer.add_scalar('Training/GanLoss/A_to_B', loss_GAN_A2B, iteration)
        writer.add_scalar('Training/GanLoss/B_to_A', loss_GAN_B2A, iteration)
        writer.add_scalar('Training/CycleLoss/ABA', loss_cycle_ABA, iteration)
        writer.add_scalar('Training/CycleLoss/BAB', loss_cycle_BAB, iteration)
        writer.add_scalar('Training/Generator/Total_loss', loss_G, iteration)



def train(opt):

    netG_A2B = Unet2(3,3)
    netG_B2A = Unet2(3,3)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    if opt.use_cuda:
        netG_A2B = netG_A2B.cuda()
        netG_B2A = netG_B2A.cuda()
        netD_A = netD_A.cuda()
        netD_B = netD_B.cuda()

    netG_A2B_optimizer = optimizer.Adam(params=netG_A2B.parameters(), lr=opt.lr,betas=(0.5, 0.999))
    netG_B2A_optimizer = optimizer.Adam(params=netG_B2A.parameters(), lr=opt.lr,betas=(0.5, 0.999))
    netD_A_optimizer = optimizer.Adam(params=netD_A.parameters(), lr=opt.lr,betas=(0.5, 0.999))
    netD_B_optimizer = optimizer.Adam(params=netD_B.parameters(), lr=opt.lr,betas=(0.5, 0.999))

    optimizers = dict()
    optimizers['G1'] = netG_A2B_optimizer
    optimizers['G2'] = netG_B2A_optimizer
    optimizers['D1'] = netD_A_optimizer
    optimizers['D2'] = netD_B_optimizer

    # Dataset loader
    transforms_ =  [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    tarindataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),batch_size=opt.batchSize, shuffle=True)

    #writer
    writer = SummaryWriter(opt.log_dir)

    for epoch in range(0, opt.n_epochs):
        for ii, batch in enumerate(tarindataloader):
            # Set model input
            real_A = Variable(batch['A'])
            real_B = Variable(batch['B'])

            if opt.use_cuda:
                real_A = real_A.cuda()
                real_B = real_B.cuda()

            train_one_step(use_cuda =opt.use_cuda,
                        netG_A2B = netG_A2B,
                        netG_B2A = netG_B2A,
                        netD_A = netD_A,
                        netD_B = netD_B,
                        real_A = real_A,
                        real_B = real_B,
                        optimizers = optimizers,
                        iteration = ii,
                       writer = writer)

            print("\nEpoch: %s Batch: %s" % (epoch, ii))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='', help='Data root',required=True)
    parser.add_argument('--use_cuda', type=int, default=1, help='use gpu to train')
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='training learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='training epoches')
    parser.add_argument('--log_dir', type=str, default='', help='log directory',required=True)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
     main()
