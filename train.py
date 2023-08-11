import os 
import sys
import argparse
import logging
import warnings 
import time 
import itertools
import random

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader
import torchvision
from tqdm import tqdm
# from tensorboardX import SummaryWriter

import utils
import datasets
import model as img_text_model
import test

from torch.cuda.amp import autocast as autocast, GradScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'fashioniq', help = "data set type")
parser.add_argument('--name', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_path', default = 'YOUR_FASHIONIQ_PATH')
parser.add_argument('--shoes_path', default = 'YOUR_SHOES_PATH')
parser.add_argument('--fashion200k', default = 'YOUR_FASHION200K_PATH')

parser.add_argument('--img_encoder', default = 'swin_large_patch4_window7_224_in22k')
parser.add_argument('--optimizer', default = 'adam')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--dropout_rate', type=float, default=0.0)

parser.add_argument('--seed', type=int, default=599)   # 6195
parser.add_argument('--learning_rate', type=float, default=3e-5)  
parser.add_argument('--lr_decay', type=int, default=20)
parser.add_argument('--lr_div', type=float, default=0.1)  
parser.add_argument('--max_decay_epoch', type=int, default=40)  

parser.add_argument('--local_t', type=float, default=7.0)   
parser.add_argument('--global_t', type=float, default=4.0)   
parser.add_argument('--use_board', type=int, default=0)

parser.add_argument('--img_weight', type=float, default=1.0)   
parser.add_argument('--class_weight', type=float, default=1.0)  
parser.add_argument('--mul_kl_weight', type=float, default=1.0)  


parser.add_argument('--model_dir', default='./experiment',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


def load_dataset():
    """Loads the input datasets."""
    print('Reading dataset ', args.dataset)
    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path = args.fashioniq_path,
            name = args.name,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.FashionIQ(
            path = args.fashioniq_path,
            name = args.name,
            split = 'val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))


    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = args.shoes_path,
            split = 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Shoes(
            path = args.shoes_path,
            split = 'test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))

    
    elif args.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=args.fashion200k,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Fashion200k(
            path=args.fashion200k,
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))

    else:
        print('Invalid dataset', args.dataset)
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset

def create_model_and_optimizer(texts):
    """Builds the model and related optimizer."""
    print('Creating model and optimizer')

    local_model = img_text_model.compose_local(texts, T=args.local_t, dropout_p=args.dropout_rate).cuda()
    global_model = img_text_model.compose_global(texts, T=args.global_t, dropout_p=args.dropout_rate).cuda()
    
    optimized_local_parameters = filter(lambda x: x.requires_grad, local_model.parameters())
    optimized_global_parameters = filter(lambda x: x.requires_grad, global_model.parameters())
    optimizer_local = torch.optim.Adam(optimized_local_parameters, lr = args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    optimizer_global = torch.optim.Adam(optimized_global_parameters, lr = args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)

    return local_model, global_model, optimizer_local, optimizer_global

def train(local_model, global_model, optimizer_local, optimizer_global, dataloader, epoch, scaler):
    """
    train on one epoch
    """
    local_model.train()
    global_model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    global_loss_avg = utils.RunningAverage()
    local_loss_avg = utils.RunningAverage()

    with tqdm(total=2 * len(dataloader)) as t:
        for i, data in enumerate(dataloader):

            if args.dataset == 'fashion200k':
                assert type(data) is list
                img1 = np.stack([d['source_img_data'] for d in data])
                img1 = torch.from_numpy(img1).float()
                img1 = torch.autograd.Variable(img1).cuda()
                img2 = np.stack([d['target_img_data'] for d in data])
                img2 = torch.from_numpy(img2).float()
                img2 = torch.autograd.Variable(img2).cuda()
                mods = [str(d['mod']['str']) for d in data]
                mods = [t.encode('utf-8').decode('utf-8') for t in mods]
            else:
                img1 = data['source_img_data'].cuda()
                img2 = data['target_img_data'].cuda()
                mods = data['mod']['str']

            optimizer_local.zero_grad()
            with autocast():
                ############### train local ###############
                loss = local_model.compute_loss(img1, mods, img2, global_model.extract_img_feature(img2)[0].detach(), global_model.extract_img_feature(img2)[1].detach(), global_model.compose_img_text(img1, mods)[0].detach())
                local_loss = args.class_weight * loss['class'] + loss['perceptual'] + args.img_weight * loss['img'] + args.mul_kl_weight * loss['mul_kl']
            scaler.scale(local_loss).backward()
            scaler.step(optimizer_local)
            scaler.update()
            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['local_loss'] = local_loss.item()
                summ.append(summary_batch)
            local_loss_avg.update(local_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(local_loss_avg()))
            t.update()
            
            optimizer_global.zero_grad()
            with autocast():
                ############### train global ###############
                loss = global_model.compute_loss(img1, mods, img2, local_model.extract_img_feature(img2)[0].detach(), local_model.extract_img_feature(img2)[1].detach(), local_model.compose_img_text(img1, mods)[1].detach())
                global_loss = args.class_weight * loss['class'] + args.img_weight * loss['img'] + args.mul_kl_weight * loss['mul_kl']
            scaler.scale(global_loss).backward()
            scaler.step(optimizer_global)
            scaler.update()

            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['global_loss'] = global_loss.item()
                summ.append(summary_batch)
            global_loss_avg.update(global_loss.item())
            t.set_postfix(loss='{:05.3f}'.format(global_loss_avg()))
            t.update()


def train_and_evaluate(local_model, global_model, optimizer_local, optimizer_global, trainset, testset, model_dir, restore_file=None):


    if args.dataset == 'fashion200k':
        trainloader = trainset.get_loader(
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers)
    else:
        trainloader = dataloader.DataLoader(trainset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args.num_workers)

    best_score = float('-inf')
    scaler = GradScaler()
    epoches = args.num_epochs
    for epoch in range(epoches):
        if epoch != 0 and epoch % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer_global.param_groups:
                g['lr'] *= args.lr_div
            for g in optimizer_local.param_groups:
                g['lr'] *= args.lr_div
        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(local_model, global_model, optimizer_local, optimizer_global, trainloader, epoch, scaler)
                
        tests = []
        for name, dataset in [('test', testset)]:
            t = test.test(args, local_model, global_model, dataset, args.dataset)
            tests += [(name + ' ' + metric_name, metric_value) for metric_name, metric_value in t]
        logging.info(tests)
        logging.info('(R10+R50) / 2 == '+str((tests[1][1]+tests[2][1])/2))
        print(tests)
        current_score = t[1][1]
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_json_path_combine = os.path.join(
                model_dir, "metrics_best.json"
            )
            test_metrics = {}
            for metric_name, metric_value in t:
                test_metrics[metric_name] = metric_value
            utils.save_dict_to_json(test_metrics, best_json_path_combine)
            torch.save(local_model, os.path.join(model_dir, 'local_model.pt'))
            torch.save(global_model, os.path.join(model_dir, 'global_model.pt'))



if __name__ == '__main__':

    # Load the parameters from json file

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')
    # fetch dataloaders

    trainset, testset = load_dataset()
    local_model, global_model, optimizer_local, optimizer_global = create_model_and_optimizer([t.encode('utf-8').decode('utf-8') for t in trainset.get_all_texts()])

    logging.info('- done.')

    # Train the model
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(local_model, global_model, optimizer_local, optimizer_global, trainset, testset, args.model_dir, args.restore_file)