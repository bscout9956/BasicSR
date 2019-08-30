import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN

####################
# ESRGAN Generator
####################

class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv', convtype='Conv2D'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale, convtype=convtype) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv', convtype='Conv2D'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA', convtype=convtype) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

####################
# PPON
####################

'''
Progressive Perception-Oriented Network for Single Image Super-Resolution
https://arxiv.org/pdf/1907.10399.pdf
'''

class PPON(nn.Module):
    def __init__(self, in_nc, nf, nb, out_nc, upscale=4, act_type='lrelu'):
        super(PPON, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)  # common
        rb_blocks = [B.RRBlock_32() for _ in range(nb)]  # L1
        LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        ssim_branch = [B.RRBlock_32() for _ in range(2)]  # SSIM
        gan_branch = [B.RRBlock_32() for _ in range(2)]  # Gan

        #upsample_block = B.upconv_block #original
        upsample_block = B.upconv_blcok #using BasicSR code

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_ssim = upsample_block(nf, nf, 3, act_type=act_type)
            upsampler_gan = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_ssim = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
            upsampler_gan = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_S = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_S = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        HR_conv0_P = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1_P = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.CFEM = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))) #Content Feature Extraction Module (CFEM)
        self.SFEM = B.sequential(*ssim_branch) #Structural Feature Extraction Module (SFEM)
        self.PFEM = B.sequential(*gan_branch) #Perceptual Feature Extraction Module (PFEM)

        self.CRM = B.sequential(*upsampler, HR_conv0, HR_conv1)  # recon l1 #content reconstruction module (CRM)
        self.SRM = B.sequential(*upsampler_ssim, HR_conv0_S, HR_conv1_S)  # recon ssim #structure reconstruction module (SRM)
        self.PRM = B.sequential(*upsampler_gan, HR_conv0_P, HR_conv1_P)  # recon gan #photo-realism reconstruction module (PRM)

    def forward(self, x):
        out_CFEM = self.CFEM(x)
        out_c = self.CRM(out_CFEM)

        out_SFEM = self.SFEM(out_CFEM)
        out_s = self.SRM(out_SFEM) + out_c

        out_PFEM = self.PFEM(out_SFEM)
        out_p = self.PRM(out_PFEM) + out_s

        return out_c, out_s, out_p


####################
# VGG Discriminator
####################


# VGG style Discriminator
class Discriminator_VGG(nn.Module):
    def __init__(self, size, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        super(Discriminator_VGG, self).__init__()

        conv_blocks = []
        conv_blocks.append(B.conv_block(  in_nc, base_nf, kernel_size=3, stride=1, norm_type=None, \
            act_type=act_type, mode=mode))
        conv_blocks.append(B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode))

        cur_size = size // 2
        cur_nc = base_nf
        while cur_size > 4:
            out_nc = cur_nc * 2 if cur_nc < 512 else cur_nc
            conv_blocks.append(B.conv_block(cur_nc, out_nc, kernel_size=3, stride=1, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            conv_blocks.append(B.conv_block(out_nc, out_nc, kernel_size=4, stride=2, norm_type=norm_type, \
                act_type=act_type, mode=mode))
            cur_nc = out_nc
            cur_size //= 2

        self.features = B.sequential(*conv_blocks)

        # classifier
        if arch=='PPON':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 128), nn.LeakyReLU(0.2, True), nn.Linear(128, 1))
        else: #arch='ESRGAN':
            self.classifier = nn.Sequential(
                nn.Linear(cur_nc * cur_size * cur_size, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


########################################
# SRPGAN-like Discriminator/Feature Extractor
########################################
#####SRPGAN
###Originally based on PatchGAN from pix2pix. Conditional GANs need to take both input and output images, need to modify to make it more similar
class TDiscriminator(nn.Module):
    def __init__(self, input_shape=(128,128), in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        """Construct a (PatchGAN) discriminator
        Parameters:
            in_nc (int)  -- the number of channels in input images
            base_nf (int)       -- the number of filters in the last conv layer
        """
        super(TDiscriminator, self).__init__()
        self.block1 = B.conv_block(in_nc, base_nf, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block2 = B.conv_block(base_nf, base_nf*2, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block3 = B.conv_block(base_nf*2, base_nf*4, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block4 = B.conv_block(base_nf*4, base_nf*8, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block5 = B.conv_block(base_nf*8, base_nf*16, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block6 = B.conv_block(base_nf*16, base_nf*32, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block7 = nn.Sequential(
            nn.Conv2d(base_nf*32, base_nf*16, kernel_size=(1, 1), stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.block8 = nn.Conv2d(base_nf*16, base_nf*8, kernel_size=(1, 1), stride=1, padding=1)
        self.block9 = nn.Sequential(
            nn.Conv2d(base_nf*8, base_nf*2, kernel_size=(1, 1), stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.block10 = nn.Sequential(
            nn.Conv2d(base_nf*2, base_nf*2, kernel_size=(3, 3), stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.block11 = B.conv_block(base_nf*2, base_nf*8, kernel_size=3, stride=1, dilation=1, norm_type=None, act_type=None, mode=mode)
        in_size = self.infer_lin_size(input_shape)

        self.out_block = nn.Sequential(
            B.Flatten(),
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )

    def infer_lin_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        model = B.sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
        )
        size = model(input).data.view(bs, -1).size(1)
        return size

    def forward(self, x):
        feature_maps = []
        
        x = self.block1(x)
        feature_maps.append(x)

        x = self.block2(x)
        feature_maps.append(x)

        x = self.block3(x)
        feature_maps.append(x)

        x = self.block4(x)
        feature_maps.append(x)

        x = self.block5(x)
        feature_maps.append(x)

        x = self.block6(x)
        feature_maps.append(x)

        x = self.block7(x)
        feature_maps.append(x)

        block8 = self.block8(x)

        x = self.block9(block8)
        feature_maps.append(x)

        x = self.block10(x)
        feature_maps.append(x)

        block11 = self.block11(x)

        final_block = nn.functional.leaky_relu(block8 + block11, 0.2)
        feature_maps.append(final_block)

        out = self.out_block(final_block)
        
        return out, feature_maps


class SRPGANDiscriminator(nn.Module):
    def __init__(self, input_shape=(128,128), in_nc=3, base_nf=64, norm_type='batch', act_type='leakyrelu', mode='CNA', convtype='Conv2D', arch='ESRGAN'):
        """Construct a (PatchGAN) discriminator
        Parameters:
            in_nc (int)  -- the number of channels in input images
            base_nf (int)       -- the number of filters in the last conv layer
        """
        super(SRPGANDiscriminator, self).__init__()
        self.block1 = B.conv_block(in_nc, base_nf, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block2 = B.conv_block(base_nf, base_nf*2, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block3 = B.conv_block(base_nf*2, base_nf*4, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block4 = B.conv_block(base_nf*4, base_nf*8, kernel_size=4, stride=2, dilation=1, norm_type=None, act_type=act_type, mode=mode)
        self.block5 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, dilation=1, norm_type=None, act_type=None, mode=mode)
        self.block6 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, dilation=1, norm_type=None, act_type=None, mode=mode)

        in_size = self.infer_lin_size(input_shape)

        self.out_block = nn.Sequential(
            B.Flatten(),
            nn.Linear(in_size, 1),
            nn.Sigmoid(),
        )

    def infer_lin_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        model = B.sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
        )
        size = model(input).data.view(bs, -1).size(1)
        return size

    def forward(self, x):
        feature_maps = []

        x = self.block1(x)
        feature_maps.append(x)

        x = self.block2(x)
        feature_maps.append(x)

        x = self.block3(x)
        feature_maps.append(x)

        x = self.block4(x)
        feature_maps.append(x)

        block5 = self.block5(x)
        block6 = self.block6(x)
        
        final_block = nn.functional.leaky_relu(block5 + block6, 0.2)
        feature_maps.append(final_block)

        out = self.out_block(final_block)
        
        return out, feature_maps

#####SRPGAN


####################
# ESRGAN Perceptual Network
####################

# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')): #PPON uses cuda instead of CPU
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device) 
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output
