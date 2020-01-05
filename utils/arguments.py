import argparse
import os, sys
import os.path as osp

def mkdirifmissing(path):
    if not osp.isdir(path):
        os.mkdir(path)

class Arguments(object):
    __feat_dims = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
    }
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        working_dir = osp.dirname(osp.abspath(__file__))
        ## Debug options
        parser.add_argument('--debug', action='store_true',help="Debug mode")
        parser.add_argument('--test-baseline', action='store_true',help="Before train")
        parser.add_argument('--show-gpu', action='store_true',help="GPU Usage")
        
        ## Training stage
        parser.add_argument('--pretrained', action='store_true',help="Continue training of this stage or not")
        parser.add_argument('--stage', type=int, default=1, choices = [1,2,3], help='different training stages')
        
        ## Experiment directory
        parser.add_argument('--exp-dir', type=str, metavar='PATH',default=osp.join(working_dir, './experiments'))
        parser.add_argument('--gen-img-dir', type=str, metavar='PATH',default=osp.join(working_dir, './experiments'))
        
        #### Data options
        # Dataset
        parser.add_argument('--data-dir', type=str, metavar='PATH',default=osp.join(working_dir, './data'))
        parser.add_argument('--dataset', type=str, default='market1501')
        parser.add_argument('--data-portion', type=float, default=1.0)
        parser.add_argument('--num-class', type=int, default=751,help='number of ids in the dataset')
        # Dataloader
        parser.add_argument('--train-batch-size', type=int, default=16, help='train batch size')
        parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
        parser.add_argument('--gen-batch-size', type=int, default=16, help='test batch size')
        parser.add_argument('--workers', default=4, type=int, help='num threads for loading data')
        parser.add_argument('--np-ratio', default=3, type=int, help='negative positive ratio for train set')
        parser.add_argument('--height', type=int, default=256, help='input image height')
        parser.add_argument('--width', type=int, default=128, help='input image width')
        parser.add_argument('--pose-aug', type=str, default='gauss')
        #### Model parameters
        # Model pretrained checkpoints
        parser.add_argument('--netE-pretrain', type=str, default='')
        parser.add_argument('--netV-pretrain', type=str, default='')
        parser.add_argument('--netG-pretrain', type=str, default='')
        parser.add_argument('--netDi-pretrain', type=str, default='')
        parser.add_argument('--netDp-pretrain', type=str, default='')
        # Encoder parameters
        parser.add_argument('--arch', type=str, default='resnet50')
        parser.add_argument('--num-parts', type = int,default = 0, help="Use part based feats")
        parser.add_argument('--part-dims', type = int,default = 256, help="Feature dimension of each part")
        parser.add_argument('--no-downsampling', action='store_true',help="Use last downsampling")
        # Verification parameters
        parser.add_argument('--classifier-type', type=str, default='siam')
        
        #GAN Loss Parameters
        parser.add_argument('--smooth-label', action='store_true',help="Use last downsampling")
        parser.add_argument('--lambda-gan-di', type = float,default = 1, help="Weight of loss")
        parser.add_argument('--lambda-gan-dp', type = float,default = 1, help="Weight of loss")
        parser.add_argument('--lambda-recon', type = float,default = 1, help="Weight of loss")
        parser.add_argument('--lambda-veri', type = float,default = 1, help="Weight of loss")
        parser.add_argument('--lambda-sp', type = float,default = 1, help="Weight of loss")

        # Generator Parameters
        parser.add_argument('--gen-net', type=str, default='cpg', choices = ['cpg','unet'], help='pick generator network')
        parser.add_argument('--drop', type=float, default=0, help="dropout parameter for cpg")
        parser.add_argument('--use-pose-conv', action='store_true',help="Use pose conv before concat")
        parser.add_argument('--pose-dim', type = int,default = 128, help="Final dimension pose vector(cpg)/channel(unet)")
        parser.add_argument('--noise-dim', type = int,default = 256, help="Final dimension noise vector(cpg)")

        
        #### Optimizer setting
        # Learning rate parameters
        parser.add_argument('--e-lr', type=float, default=5e-4, help='initial learning rate')
        parser.add_argument('--v-lr', type=float, default=5e-4, help='initial learning rate')
        parser.add_argument('--pcb-lr', type=float, default=0.1, help="learning rate of pcb")
        parser.add_argument('--g-lr', type=float, default=0.1, help="learning rate of pcb")
        parser.add_argument('--di-lr', type=float, default=0.1, help="learning rate of pcb")
        parser.add_argument('--dp-lr', type=float, default=0.1, help="learning rate of pcb")

        parser.add_argument('--momentum', type=float, default=0.1, help="momentum for sgd optimizer")
        parser.add_argument('--weight-decay', type=float, default=0.1, help="weight decay for sgd optimizer")
        parser.add_argument('--epochs', type=int, default=50, help='How many epochs to run the model')
        # Scheduler parameters
        parser.add_argument('--lr-step', type=int, default=20, help='# of iter to lower learning rate by a factor of 0.1')
        #### Save/Eval Steps
        parser.add_argument('--save-step', type=int, default=3, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--eval-step', type=int, default=3, help='frequency of evaluate checkpoints at the end of epochs')
        
        self.args = parser.parse_args()
        ## Create missing directories
        mkdirifmissing(self.args.exp_dir)
        mkdirifmissing(self.args.gen_img_dir)
        self.args.log_dir = osp.join(self.args.exp_dir,'train.log')
        self.args.summary_dir = osp.join(self.args.exp_dir,'summaries')
        mkdirifmissing(self.args.summary_dir)
        self.args.ckpt_dir = osp.join(self.args.exp_dir,'ckpt')
        mkdirifmissing(self.args.ckpt_dir)
        
        ## Calculate embedding dimension
        if self.args.num_parts == 0:
            self.args.embed_dim = Arguments.__feat_dims[self.args.arch]
        else:
            self.args.embed_dim = self.args.num_parts * self.args.part_dims
        self.show_args()

    def parse(self):
        return self.args

    def show_args(self):
        args = vars(self.args)
        fpath = osp.join(self.args.exp_dir,'arguments.txt')
        with open(fpath, 'w') as f:
            f.write('----------- Arguments ------------\n')
            for k, v in sorted(args.items()):
                msg = '%s: %s \n' % (str(k), str(v))
                f.write(msg)
                print(msg)
            f.write('-------------- End ---------------')
