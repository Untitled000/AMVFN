
import argparse
import os
import logging

import torch
import torch.nn as nn

from model.MultiViewModel import AttentionMultiViewFusionNet

from loss.loss import MutualDistillationLoss
from engine.trainer import TrainerEngine
from engine.evaluator import Evaluator
import numpy as np
from my_dataset import ImagePairDataset
from utils import str2bool

def main(logger):

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a dataset object
    trainval_dataset = ImagePairDataset(args.data_dir, ds_type=1)
    # Split the dataset into train and validation
    dataset_train, dataset_val = torch.utils.data.random_split(trainval_dataset, [int(0.8*len(trainval_dataset)), int(0.2*len(trainval_dataset))])
    # Create dataloader objects
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # Create a model object
    model = AttentionMultiViewFusionNet(args.architecture, num_classes=args.num_classes, n=args.n)
    model.to(args.device)

    print('Number of classes: ', args.num_classes)

    optimizer_model = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer_model, max_lr=args.lr,
                                                    epochs=args.num_epochs,
                                                    div_factor=10,
                                                    steps_per_epoch=len(dataset_train) // args.batch_size,
                                                    final_div_factor=1000,
                                                    pct_start=5 / args.num_epochs, anneal_strategy='cos')

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if args.use_mutual_distillation_loss:
        md_loss = MutualDistillationLoss(temp=args.md_temp, lambda_hyperparam=args.md_lambda)
    else:
        md_loss = None

    evaluator = Evaluator(model=model, n=args.n)
    trainer = TrainerEngine(model=model, lr_scheduler_model=scheduler, criterion=criterion, optimizer_model=optimizer_model, 
                            evaluator=evaluator, md_loss=md_loss, grad_clip_norm=args.grad_clip_norm, logger=logger,
                            save_dir=args.save_dir)
    # Train the model
    trainer.train(loader_train, args.num_epochs, loader_val)
    logger.info('Training done!\n')
    # Load the best model
    best_weights = torch.load(f'{args.save_dir}/best.pth')
    model.load_state_dict(best_weights['model'])
    logger.info('Evaluating on test set:\n')

    # Create a dataset object
    dataset_test = ImagePairDataset(args.data_dir, ds_type=0)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # Evaluate the model
    score_dict = evaluator.evaluate(loader_test)
    for view_type in score_dict:
        for metric in score_dict[view_type]:
            logger.info(f'Test {view_type} {metric}: {score_dict[view_type][metric]}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Multiview Model Training", allow_abbrev=False
    )
    parser.add_argument("--device", default='cuda:2', type=str, help="device")
    parser.add_argument("--data_dir", default='/data1/wcy/data', type=str, help="location to images")
    parser.add_argument("--architecture", default='vit_tiny_r_s16_p8_224', type=str, help="model architecture")
    parser.add_argument('--pretrained_weights', default=False, type=str2bool, help='use pretrained weights')
    parser.add_argument('--save_dir', default='output_final_2/exp_9', type=str, help='save location for model weights and log')
    parser.add_argument('--seed', default=3407, type=int, help='seed')
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--momentum", default=.9, type=float, help="momentum")
    parser.add_argument("--num_epochs", default=20, type=int, help="number of epochs for training")
    parser.add_argument("--num_classes", default=2, type=int, help='the number of categories')
    parser.add_argument("--n", default=2, type=int, help="number of images per input")
    parser.add_argument("--num_workers", default=8, type=int, help="num dataloading workers")
    parser.add_argument('--use_mutual_distillation_loss', default=True, type=str2bool, help='use mutual distillation loss')
    parser.add_argument("--md_temp", default=4., type=float, help='mutual distillation temperature')
    parser.add_argument("--md_lambda", default=.1, type=float, help='mutual distillation temperature lambda hyperparm')
    parser.add_argument("--grad_clip_norm", default=80., type=float, help='grad clip norm value')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logfile = f'{args.save_dir}/log.txt'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    for param in vars(args):
        logger.info(f'{param}: {getattr(args, param)}')

    PARAMS = {}
    for arg in vars(args):
        PARAMS[arg] = getattr(args, arg)

    # args.n = args.num_images
    main(logger=logger)