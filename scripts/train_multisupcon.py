import os
import argparse
import time
import umap
import math
import torch.nn as nn
# import wandb
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet18
from losses import MultiSupConLoss
from utils import AverageMeter, adjust_learning_rate, warmup_learning_rate
from datasets import miniRINmultilabel, TwoCropTransform

def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch Multi supervised contrastive training')
    
    #############################
    # data and model parameters #
    #############################
    parser.add_argument('--data', type=str, default='../../data/celebA')
    parser.add_argument('--data-name', type=str, default='CELEBA')
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--num-classes', type=int, default=80)
    parser.add_argument('--image-size', default=224, type=int,
                        metavar='N', help='input image size (default: 448)')
    
    ###############################
    # MultiSupCon specific params #
    ###############################
    parser.add_argument('--method', type=str, default='MultiSupCon',
                        choices=['MultiSupCon', 'SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--feat-dim', type=int, default=128,
                        help='feature dimension for contrastive learning')
    parser.add_argument('--c_treshold', type=float, default=0.3,
                        help='Jaccard sim split parameter')
    
    ###############################
    ####### optim parameters ######
    ###############################
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    
    ###############################
    ####### other parameters ######
    ###############################
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--vis_3d', default=True, type=bool,
                        metavar='N', help='Visualize in 3d')
    parser.add_argument('--run', default=0, type=int,
                        metavar='N', help='run number')
    parser.add_argument("--dump_path", type=str, default="../experiment",
                    help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--checkpoint_freq", type=int, default=1, help="checkpoint frequency")
    
    return parser


def main():
    # Prepering environment
    args = parse_option().parse_args()
    # fix_random_seeds(args.seed)  # 必要なら乱数シードだけ
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train_dataset = miniRINmultilabel(transform=TwoCropTransform(transforms.Compose([
                                transforms.RandomResizedCrop(size=args.image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.ToTensor(),
                                ])))
   
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError(f"Dataset is empty. Please check the data path: {args.data}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,  # シングルプロセス
        pin_memory=True,
    )
    
    # Build model
    model = resnet18()
    # 1チャンネル対応に変更
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.cuda()
    cudnn.benchmark = True
    
    # Build optimizer and criterion
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
        
    # Warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 3 # epoch100にするときは10に変更
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    
    criterion = MultiSupConLoss(temperature=args.temp, c_treshold=args.c_treshold)
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    # Check for the checkpoints
    log_path = os.path.join(args.dump_path, f"run_{args.run}")
    if os.path.isdir(log_path):
        resume = True
    else:
        resume = False
        os.makedirs(log_path, exist_ok=True)
    
    # Load checkpoint
    resume = False
    if resume:
        # Get last restore
        checkpoint_last = os.path.join(log_path, "last_checkpoint.pth.tar")
        checkpoint_best = os.path.join(log_path, "best_checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_last,
               map_location="cuda:" + str(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best = torch.load(checkpoint_best,
            map_location="cuda:" + str(torch.cuda.current_device()))
    else:
        start_epoch = 1
        best = {}
    
    ###############################
    ########### TRAINING ##########
    ###############################
    for epoch in range(start_epoch, args.epochs+1):
        
        # train the network for one epoch
        print(f"============ Starting epoch {epoch} ... ============")
        # エポック計測開始（GPU同期）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.time()
        
        adjust_learning_rate(args, optimizer, epoch)
        # train the network
        scores = train(
            train_loader,
            model,
            optimizer,
            criterion,
            epoch,
            args
        )
        
        # エポック計測終了（GPU同期）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} training time: {epoch_time:.3f} sec ({epoch_time/60:.2f} min)")
        
        # save checkpoints
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(log_path, f"{epoch}_checkpoint.pth.tar")
            checkpoint_last = os.path.join(log_path, "last_checkpoint.pth.tar")
            torch.save({ 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': scores[1],
                }, checkpoint_path)
            torch.save({ 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': scores[1],
                }, checkpoint_last)

    ###############################
    ########### Finished ##########
    ###############################
    print("============ Finished ============")
    print(f"Best loss: {best['loss']} on epoch {best['epoch']}")


def train(train_loader, model, optimizer, criterion, epoch, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    scaler = GradScaler()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        print(images[0].shape)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        #labels = labels.max(dim=1)[0]
        bsz = labels.shape[0]
        
        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        # compute loss
        with autocast():  # mixed precision
            features = model(images).float()
        
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        f1 = torch.nn.functional.normalize(f1, dim=1)
        f2 = torch.nn.functional.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    
        if args.method == 'MultiSupCon':
            loss = criterion(features, labels)
        elif args.method == 'SupCon':
            loss = criterion(features, labels, multi=False)
        elif args.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                            format(args.method))
        
        # update metric
        losses.update(loss.item(), bsz)
        print(loss.item())

        # Optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    return (epoch, losses.avg)

def validate(train_loader, model, vis_3d=True):
    model.eval()
    outputs = []
    labels = []
    
    # Initialize umap
    if vis_3d:
        reducer = umap.UMAP(n_components=3)
    else:
        reducer = umap.UMAP(n_components=2)
    
    # Start validating
    with torch.no_grad():
        for idx, (images, label) in enumerate(train_loader):
            images = images[0]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)

            # compute loss
            with autocast():
                outputs.append(
                    torch.nn.functional.normalize(model(images), dim=1).cpu().detach())
            labels.append(label.max(dim=1)[0].cpu().detach())
    
    # Do dimension reduction
    embedding = reducer.fit_transform(torch.cat(outputs).numpy())
    labels = torch.cat(labels)
    
    # Create text for embeding
    results = []
    for label in labels:
        result = torch.nonzero(label).reshape(-1)
        if result.shape[0] > 1:
            results.append(str(np.array2string(result.numpy(), separator=',')))
        else:
            results.append(str(result.numpy()[0]))
    
    # Create dataframe
    if vis_3d:
        df = pd.DataFrame({
            'x': embedding[:,0],
            'y': embedding[:,1],
            'z': embedding[:,2],
            'label': results,
            }
        )
        # Create scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', hover_data=
                dict({'x': False, 'y': False, 'z': False, 'label': True}))
        # Tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    else:
        df = pd.DataFrame({
            'x': embedding[:,0],
            'y': embedding[:,1],
            'label': result,
            }
        )
        # Create scatter plot
        fig = px.scatter(df, x='x', y='y', hover_data=
                dict({'x': False, 'y': False, 'label': True}))
        
    return fig


if __name__ == '__main__':
    main()
