import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def create_generator_net(args):
    if args.gen_net == 'unet':
        if args.num_parts != 0:e_dim = args.part_dims
        else: e_dim = args.embed_dim
        return UNet(embed_dim=e_dim,
                    pose_dim=args.pose_dim,
                    noise_dim=args.noise_dim,
                    bilinear=True,
                    part=args.num_parts!=0,
                    use_pose_conv=args.use_pose_conv,
                    dropout=args.drop)
                              
    elif args.gen_net == 'cpg':
        return CustomPoseGenerator(pose_feature_nc=args.pose_dim,
                                   reid_feature_nc=args.embed_dim, 
                                   noise_nc = args.noise_dim, 
                                   pose_nc = 18, 
                                   output_nc = 3, 
                                   dropout = args.drop, 
                                   height = args.height)
    else: raise ValueError("Incorrect argument for generator")

#################################################################################
######################### UNET GENERATOR  ################################
#################################################################################
class ResBlock(nn.Module):
    def __init__(self,channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1)
        self.norm = nn.BatchNorm2d(channel)
        
    def forward(self,x):
        res = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = res + x
        x = self.norm(x)
        return x

class UNetDown(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNetDown, self).__init__()
        self.down_conv = nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=2,padding=1)
        self.res = ResBlock(out_ch)

    def forward(self,x):
        x = self.down_conv(x)
        x = self.res(x)
        return x

class UNetUp(nn.Module):
    def __init__(self,concat_ch,out_ch,activation = 'relu',conv_t = False,dropout=0):
        super(UNetUp, self).__init__()
        self.res = ResBlock(concat_ch)
        if conv_t:
            self.up = nn.ConvTranspose2d(concat_ch, concat_ch,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=True)
        else:
            self.up = nn.Upsample(scale_factor = 2, mode='bilinear',align_corners = True)
        self.out_conv = nn.Conv2d(concat_ch,out_ch,kernel_size=1,stride=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:self.activation=None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,e):
        x = torch.cat([x,e],dim=1)
        x = self.up(x)
        x = self.res(x)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(self,embed_dim=512,pose_dim=18,noise_dim=36,bilinear=True,part=True,use_pose_conv=False,dropout=0):
        print("Using UNet")
        super(UNet, self).__init__()
        self.pose_dim = pose_dim
        self.noise_dim = noise_dim
        self.part = part
        self.use_pose_conv = use_pose_conv
        
        if self.part:self.small_h=24
        else:self.small_h = 16
        print("AUTOMATE SMALLL_H BY GIVING WIDTH TO THE UNet")
        if self.use_pose_conv:
            self.pose_conv = nn.Sequential(
                        nn.Conv2d(18,64,kernel_size = 3,stride=1,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,self.pose_dim,kernel_size = 3,stride=1,padding=1),
                        nn.ReLU(),
                        )
        self.down0 = UNetDown(embed_dim+pose_dim+noise_dim,64)
        self.down1 = UNetDown(64,128)
        self.down2 = UNetDown(128,256)
        self.down3 = UNetDown(256,384)
        self.fc_down = nn.Linear(384*self.small_h*8,64)
        
        self.fc_up = nn.Linear(64,64*self.small_h*8)
        self.up0 = UNetUp(384+64,192,dropout=dropout)
        self.up1 = UNetUp(256+192,128,dropout=dropout)
        self.up2 = UNetUp(128+128,64,dropout=dropout)
        self.img_up = UNetUp(64+64,3,activation = None,dropout=0)
        
    def tile_fdgan(self,features):
        return features.repeat(1,1,256,128)

    def tile_part(self,features,use_conv=False):
        if use_conv:
            f = self.part_tile_convs(features)
            return f
        b = features.size(0)
        f = features.view(b,6,256,1).permute(0,2,1,3)
        f = f.view(b,256,6,1,1)
        f = f.repeat(1,1,1,64,128)
        f = f.view(b,256,384,128)
        return f

    def inspect_layer(self,x,str_x):
        #print(str_x+ " shape:",x.shape)
        #print(str_x+ " max:",torch.max(x).data)
        #print(str_x+ " min:",torch.min(x).data)
        #print(str_x+ " mean:",torch.mean(x).data)
        pass
    
    def forward(self,pose,features,noise):
        if self.part:
            f = self.tile_part(features)
        else:
            f = self.tile_fdgan(features)
        if self.use_pose_conv:
            self.inspect_layer(pose,"before pose")
            pose = self.pose_conv(pose)
            self.inspect_layer(pose,"after pose")
        self.inspect_layer(f,"f")
        self.inspect_layer(pose,"pose")
        
        ## Create noise
        if self.noise_dim>0:
            noise = torch.randn(features.size(0),self.noise_dim,384,128)
            noise = Variable(noise).cuda()
            self.inspect_layer(noise,"noise")
            inp = torch.cat([pose,f,noise],dim = 1)
        else:inp = torch.cat([pose,f],dim = 1)
        self.inspect_layer(inp,"inp")
        e0 = self.down0(inp) 
        self.inspect_layer(e0,"e0") ## (b,64,128,64)
        e1 = self.down1(e0) 
        self.inspect_layer(e1,"e1") ## (b,128,64,32)
        e2 = self.down2(e1) 
        self.inspect_layer(e2,"e2") ## (b,256,32,16)
        e3 = self.down3(e2) 
        self.inspect_layer(e3,"e3") ## (b,512,16,8)
        z1 = e3.view(e3.size(0),-1)
        self.inspect_layer(z1,"z1") ## (b,512*16*8)
        z2 = self.fc_down(z1)
        self.inspect_layer(z2,"z2") ## (b,64)
        z3 = self.fc_up(z2)
        self.inspect_layer(z3,"z3") ## (b,64*16*8)
        x = z3.view(z3.size(0),64,self.small_h,8)
        self.inspect_layer(x,"x") ## (b,64,16,8)
        u0 = self.up0(x,e3)
        self.inspect_layer(u0,"u0") ## (b,192,32,16)
        u1 = self.up1(u0,e2)
        self.inspect_layer(u1,"u1") ## (b,128,64,32)
        u2 = self.up2(u1,e1)
        self.inspect_layer(u2,"u2") ## (b,64,128,64)
        img = self.img_up(u2,e0)
        self.inspect_layer(img,"img") ## (b,3,256,128)
        assert img.shape[1] == 3
        assert img.shape[2:] == img.shape[2:]
        return img


#################################################################################
######################### CUSTOM POSE GENERATOR  ################################
#################################################################################

class CustomPoseGenerator(nn.Module):
    def __init__(self, pose_feature_nc, reid_feature_nc, noise_nc, pose_nc=18, output_nc=3, 
                        dropout=0.0, norm_layer=nn.BatchNorm2d, fuse_mode='cat', connect_layers=0,height = 256):
        super(CustomPoseGenerator, self).__init__()
        print("Using CustomPoseGenerator")
        assert (connect_layers>=0 and connect_layers<=5)
        ngf = 64
        self.connect_layers = connect_layers
        self.fuse_mode = fuse_mode
        self.norm_layer = norm_layer
        self.dropout = dropout

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d

        input_channel = [[8, 8, 4, 2, 1],
                        [16, 8, 4, 2, 1],
                        [16, 16, 4, 2, 1],
                        [16, 16, 8, 2, 1],
                        [16, 16, 8, 4, 1],
                        [16, 16, 8, 4, 2]]

        ##################### Encoder #########################
        self.en_conv1 = nn.Conv2d(pose_nc, ngf, kernel_size=4, stride=2, padding=1, bias=self.use_bias)
        # N*64*128*64
        self.en_conv2 = self._make_layer_encode(ngf, ngf*2)
        # N*128*64*32
        self.en_conv3 = self._make_layer_encode(ngf*2, ngf*4)
        # N*256*32*16
        self.en_conv4 = self._make_layer_encode(ngf*4, ngf*8)
        # N*512*16*8
        self.en_conv5 = self._make_layer_encode(ngf*8, ngf*8)
        # N*512*8*4
        k_height = 8
        if height == 384:
            k_height = 12
            reid_feature_nc = 1536

        en_avg = [nn.LeakyReLU(0.2, True),
                  nn.Conv2d(ngf * 8, pose_feature_nc,
                    kernel_size=(k_height,4), bias=self.use_bias),
                  norm_layer(pose_feature_nc)]
        self.en_avg = nn.Sequential(*en_avg)
        # N*512*1*1

        ##################### Decoder #########################
        if fuse_mode=='cat':
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(pose_feature_nc+reid_feature_nc+noise_nc, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        elif fuse_mode=='add':
            nc = max(pose_feature_nc, reid_feature_nc, noise_nc)
            self.W_pose = nn.Linear(pose_feature_nc, nc, bias=False)
            self.W_reid = nn.Linear(reid_feature_nc, nc, bias=False)
            self.W_noise = nn.Linear(noise_nc, nc, bias=False)
            de_avg = [nn.ReLU(True),
                    nn.ConvTranspose2d(nc, ngf * 8,
                        kernel_size=(8,4), bias=self.use_bias),
                    norm_layer(ngf * 8),
                    nn.Dropout(dropout)]
        else:
            raise ('Wrong fuse mode, please select from [cat|add]')
        self.de_avg = nn.Sequential(*de_avg)
        # N*512*8*4

        self.de_conv5 = self._make_layer_decode(ngf * input_channel[connect_layers][0],ngf * 8)
        # N*512*16*8
        self.de_conv4 = self._make_layer_decode(ngf * input_channel[connect_layers][1],ngf * 4)
        # N*256*32*16
        self.de_conv3 = self._make_layer_decode(ngf * input_channel[connect_layers][2],ngf * 2)
        # N*128*64*32
        self.de_conv2 = self._make_layer_decode(ngf * input_channel[connect_layers][3],ngf)
        # N*64*128*64
        de_conv1 = [nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * input_channel[connect_layers][4],output_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                    nn.Tanh()]
        self.de_conv1 = nn.Sequential(*de_conv1)
        # N*3*256*128

    def _make_layer_encode(self, in_nc, out_nc):
        block = [nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_nc, out_nc,
                        kernel_size=4, stride=2,
                        padding=1, bias=self.use_bias),
                self.norm_layer(out_nc)]
        return nn.Sequential(*block)

    def _make_layer_decode(self, in_nc, out_nc):
        block = [nn.ReLU(True),
                nn.ConvTranspose2d(in_nc, out_nc,
                    kernel_size=4, stride=2,
                    padding=1, bias=self.use_bias),
                self.norm_layer(out_nc),
                nn.Dropout(self.dropout)]
        return nn.Sequential(*block)

    def decode(self, model, fake_feature, pose_feature, cnlayers):
        if cnlayers>0:
            return model(torch.cat((fake_feature,pose_feature),dim=1)), cnlayers-1
        else:
            return model(fake_feature), cnlayers

    def forward(self, posemap, reid_feature, noise):
        batch_size = posemap.data.size(0)

        pose_feature_1 = self.en_conv1(posemap)
        pose_feature_2 = self.en_conv2(pose_feature_1)
        pose_feature_3 = self.en_conv3(pose_feature_2)
        pose_feature_4 = self.en_conv4(pose_feature_3)
        pose_feature_5 = self.en_conv5(pose_feature_4)
        pose_feature = self.en_avg(pose_feature_5)
        #print("pose f",pose_feature.shape)

        if self.fuse_mode=='cat':
            #print("r",reid_feature.shape)
            #print("p",pose_feature.shape)
            #print("n",noise.shape)
            print("reid",reid_feature.shape)
            print("pose_feature",pose_feature.shape)
            print("noise",noise.shape)
            feature = torch.cat((reid_feature, pose_feature, noise),dim=1)
        elif self.fuse_mode=='add':
            feature = self.W_reid(reid_feature.view(batch_size, -1)) + \
                    self.W_pose(pose_feature.view(batch_size, -1)) + \
                    self.W_noise(noise.view(batch_size,-1))
            feature = feature.view(batch_size,-1,1,1)

        fake_feature = self.de_avg(feature)

        cnlayers = self.connect_layers
        fake_feature_5, cnlayers = self.decode(self.de_conv5, fake_feature, pose_feature_5, cnlayers)
        fake_feature_4, cnlayers = self.decode(self.de_conv4, fake_feature_5, pose_feature_4, cnlayers)
        fake_feature_3, cnlayers = self.decode(self.de_conv3, fake_feature_4, pose_feature_3, cnlayers)
        fake_feature_2, cnlayers = self.decode(self.de_conv2, fake_feature_3, pose_feature_2, cnlayers)
        fake_feature_1, cnlayers = self.decode(self.de_conv1, fake_feature_2, pose_feature_1, cnlayers)

        fake_imgs = fake_feature_1
        #print("out",fake_imgs.shape)
        return fake_imgs
