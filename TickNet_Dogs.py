# -*- coding: utf-8 -*-

import argparse
import csv
import sys
import os
# import inspect
import time
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets

# from pathlib import Path
# sys.path.append(str(Path('.').absolute().parent))
from models.datasets import *
from models.TickNet import *
import writeLogAcc as wA


def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='TickNet training script for cifar and StanfordDogs datasets.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--data-root', type=str,
                        default='../../../datasets/StanfordDogs', help='Dataset root path.')
    # parser.add_argument('-d', '--dataset', choices=['cifar10', 'cifar100', 'dogs'], required=True, help='Dataset name.')
    parser.add_argument('-d', '--dataset', type=str, choices=[
                        'cifar10', 'cifar100', 'dogs'], default='dogs', help='Dataset name.')
    parser.add_argument('--architecture-types', nargs='+', default=[
                        'basic', 'small', 'large'], help='List of architecture types to use.')
    parser.add_argument('--download', action='store_true',
                        help='Download the specified dataset before running the training.')
    parser.add_argument('-g', '--gpu-id', default=1, type=int,
                        help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=64,
                        type=int, help='Batch size.')
    parser.add_argument('-e', '--epochs', default=200,
                        type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1,
                        type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[
                        100, 150, 180], type=int, help='Learning rate schedule (epochs after which the learning rate should be dropped).')
    parser.add_argument('-m', '--momentum', default=0.9,
                        type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4,
                        type=float, help='SGD weight decay.')
    # Add the base directory argument
    parser.add_argument('--base-dir', type=str, default='.',
                        help='Base directory for saving checkpoints and reports.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--network-type', type=str, choices=[
                        'tickNet', 'spatialTickNet'], default='tickNet', help='Type of network to use.')

    return parser.parse_args()


def get_device(args):
    """
    Determine the device to use for the given arguments, including MPS for Mac Silicon.
    """
    if args.gpu_id >= 0 and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(args.gpu_id))
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_data_loader(args, train):
    """
    Return the data loader for the given arguments.
    """
    if args.dataset in ('cifar10', 'cifar100'):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])

        # cifar10 vs. cifar100
        if args.dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100

    elif args.dataset in ('dogs',):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])

        # dataset_class = models.datasets.StanfordDogs
        dataset_class = StanfordDogs

    else:
        raise NotImplementedError(
            'Can\'t determine data loader for dataset \'{}\''.format(args.dataset))

    # trigger download only once
    if args.download:
        dataset_class(root=args.data_root, train=train,
                      download=True, transform=transform)

    # instantiate dataset class and create data loader from it
    dataset = dataset_class(root=args.data_root, train=train,
                            download=False, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True if train else False, num_workers=args.workers)


def calculate_accuracy(output, target):
    """
    Top-1 classification accuracy.
    """
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    """
    Run one epoch. If `train` is `True` perform training, otherwise validate.
    """
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    batch_count = len(data_loader)
    losses = []
    accs = []
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        # record loss and measure accuracy
        loss_item = loss.item()
        losses.append(loss_item)
        acc = calculate_accuracy(output, target)
        accs.append(acc)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (n_batch % 10) == 0:
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format('train' if train else ' val ', n_epoch + 1,
                  args.epochs, n_batch + 1, batch_count, "train" if train else "val", loss_item, "train" if train else "val", 100.0 * acc))

    return (sum(losses) / len(losses), sum(accs) / len(accs))


def get_unique_file_path(result_dir, base_filename, total_epochs, batch_size):
    """
    Create a unique file path by adding an index at the end of the file name if the file already exists.
    """
    result_file_path = os.path.join(
        result_dir, f'{base_filename}_epochs{total_epochs}_batch{batch_size}.csv'
    )
    index = 1
    while os.path.isfile(result_file_path):
        result_file_path = os.path.join(
            result_dir, f'{base_filename}_epochs{total_epochs}_batch{batch_size}_{index}.csv'
        )
        index += 1
    return result_file_path


def initialize_csv_file(result_dir, total_epochs, batch_size):
    """
    Create a CSV file with a unique name and write the header.
    """
    base_filename = 'result'
    result_file_path = get_unique_file_path(
        result_dir, base_filename, total_epochs, batch_size
    )
    header = ['Epoch', 'Train Loss', 'Train Accuracy',
              'Validation Loss', 'Validation Accuracy']

    # Create directory if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    return result_file_path


def log_results_to_csv(result_file_path, current_epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    Log the results to a CSV file.
    """
    result_row = [current_epoch + 1, train_loss,
                  train_accuracy, val_loss, val_accuracy]

    with open(result_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_row)


def main():
    """
    Run the complete model training.
    """
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    device = get_device(args)
    print('Using device {}'.format(device))

    # print model with parameter and FLOPs counts
    torch.autograd.set_detect_anomaly(True)

    # Set the base directory
    arr_architecture_types = args.architecture_types
    for typesize in arr_architecture_types:
        if args.network_type == 'tickNet':
            strmode = f'StanfordDogs_TickNet_{typesize}_SE'
        elif args.network_type == 'spatialTickNet':
            strmode = f'StanfordDogs_TickNet_spatial_{typesize}_SE'

        if args.network_type == 'tickNet':
            pathout = f'{args.base_dir}/checkpoints/{strmode}'
        elif args.network_type == 'spatialTickNet':
            pathout = f'{args.base_dir}/checkpoints/spatial_{strmode}'

        filenameLOG = pathout + '/' + strmode + '.txt'
        if not os.path.exists(pathout):
            os.makedirs(pathout)

        # Directory for logging results to CSV
        if args.network_type == 'tickNet':
            result_dir = f'{args.base_dir}/report/StanfordDogs_{typesize}'
        elif args.network_type == 'spatialTickNet':
            result_dir = f'{args.base_dir}/report/StanfordDogs_spatial{typesize}'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Initialize CSV file once for the entire run
        result_file_path = initialize_csv_file(
            result_dir, args.epochs, args.batch_size
        )

        # get model
        if args.network_type == 'tickNet':
            model = build_TickNet(120, typesize=typesize, cifar=False)
        elif args.network_type == 'spatialTickNet':
            model = build_SpatialTickNet(120, typesize=typesize, cifar=False)

        model = model.to(device)

        print(model)
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

        # define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=args.schedule,
            gamma=0.1
        )

        # get train and val data loaders
        train_loader = get_data_loader(args=args, train=True)
        val_loader = get_data_loader(args=args, train=False)

        if args.evaluate:
            if args.network_type == 'tickNet':
                pathcheckpoint = f'{args.base_dir}/checkpoints/StanfordDogs/{strmode}/model_best.pth'
            elif args.network_type == 'spatialTickNet':
                pathcheckpoint = f'{args.base_dir}/checkpoints/StanfordDogs_spatial/{strmode}/model_best.pth'

            if os.path.isfile(pathcheckpoint):
                print("=> loading checkpoint '{}'".format(pathcheckpoint))
                checkpoint = torch.load(pathcheckpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(pathcheckpoint))
                return
            m = time.time()
            (val_loss, val_accuracy) = run_epoch(train=False, data_loader=val_loader, model=model,
                                                 criterion=criterion, optimizer=None, n_epoch=0, args=args, device=device)
            print(
                f'[ validating: ], loss_val={val_loss:.5f}, acc_val={100.0 * val_accuracy:.2f}%'
            )
            n = time.time()
            print((n-m)/3600)
            return

        # for each epoch...
        val_accuracy_max = None
        val_accuracy_argmax = None
        for current_epoch in range(args.epochs):
            current_learning_rate = optimizer.param_groups[0]['lr']
            print(
                f'Starting epoch {current_epoch + 1}/{args.epochs}, learning_rate={current_learning_rate}'
            )

            # train
            (train_loss, train_accuracy) = run_epoch(train=True, data_loader=train_loader, model=model,
                                                     criterion=criterion, optimizer=optimizer, n_epoch=current_epoch, args=args, device=device)

            # validate
            (val_loss, val_accuracy) = run_epoch(train=False, data_loader=val_loader, model=model,
                                                 criterion=criterion, optimizer=None, n_epoch=current_epoch, args=args, device=device)
            if (val_accuracy_max is None) or (val_accuracy > val_accuracy_max):
                val_accuracy_max = val_accuracy
                val_accuracy_argmax = current_epoch
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    f'{pathout}/checkpoint_epoch{current_epoch + 1:>04d}_{100.0 * val_accuracy_max:.2f}.pth'
                )

            # adjust learning rate
            scheduler.step()

            # save the model weights
            # torch.save({"model_state_dict": model.state_dict()}, 'checkpoint_epoch{:>04d}.pth'.format(n_epoch + 1))

            # print epoch summary
            line = (
                f'Epoch {current_epoch + 1}/{args.epochs} summary: '
                f'loss_train={train_loss:.5f}, '
                f'acc_train={100.0 * train_accuracy:.2f}%, '
                f'loss_val={val_loss:.2f}, '
                f'acc_val={100.0 * val_accuracy:.2f}% '
                f'(best: {100.0 * val_accuracy_max:.2f}% @ epoch {val_accuracy_argmax + 1})'
            )

            print('=' * len(line))
            print(line)
            print('=' * len(line))
            wA.writeLogAcc(filenameLOG, line)

            # Log results to CSV
            log_results_to_csv(result_file_path, current_epoch,
                               train_loss, train_accuracy, val_loss, val_accuracy)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)
