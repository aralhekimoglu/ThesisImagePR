import os.path as osp

from utils.data import transforms as T
from utils.data.preprocessor import Preprocessor
from utils.data.sampler import RandomPairSampler
from utils.data.dataset import Dataset

from torch.utils.data import DataLoader

def get_data(args):
    root = osp.join(args.data_dir, args.dataset)
    
    dataset = Dataset(root, split_id=0,data_portion = args.data_portion)
    
    ## Set up transformations
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(args.height, args.width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer, 
    ])

    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalizer,
    ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                    root=dataset.images_dir, transform=test_transformer),
        batch_size=args.test_batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=False)
        
    ## Set up dataloaders
    if args.stage == 1:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                        transform=train_transformer),
            sampler=RandomPairSampler(dataset.train, neg_pos_ratio=args.np_ratio),
            batch_size=args.train_batch_size, num_workers=args.workers, pin_memory=False)
        
        return dataset,train_loader,test_loader

    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                        transform=train_transformer,with_pose=True,
                        pose_root=dataset.poses_dir,pose_aug=args.pose_aug,
                        height=args.height,width=args.width,
                        pid_imgs=dataset.train_query),
            sampler=RandomPairSampler(dataset.train, neg_pos_ratio=args.np_ratio),
            batch_size=args.train_batch_size, num_workers=args.workers, pin_memory=False)
        """
        ## For training data generator
        generator_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.images_dir,
                        transform=test_transformer,with_pose=True,
                        pose_root=dataset.poses_dir,pose_aug=args.pose_aug,
                        height=args.height,width=args.width,
                        pid_imgs=dataset.train_query),
            sampler=RandomPairSampler(dataset.train, neg_pos_ratio=0),
            batch_size=args.gen_batch_size, num_workers=args.workers, pin_memory=False)
        """
        generator_loader = DataLoader(
            Preprocessor(dataset.query, root=dataset.images_dir,
                        transform=test_transformer,with_pose=True,
                        pose_root=dataset.poses_dir,pose_aug=args.pose_aug,
                        height=args.height,width=args.width,
                        pid_imgs=dataset.query_query),
            sampler=RandomPairSampler(dataset.query, neg_pos_ratio=0),
            batch_size=args.gen_batch_size, num_workers=args.workers, pin_memory=False)
        
        return dataset,train_loader,test_loader,generator_loader