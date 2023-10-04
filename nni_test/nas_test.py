import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
import nni.retiarii.strategy as strategy
from model_search import Network


import nni

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

from operations import *

import argparse

import time


parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='which dataset: cifar10, mnist, emnist, fashion, svhn, stl10, devanagari')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-4, help='min learning rate')
parser.add_argument('--lr_power_annealing_exponent_order', type=float, default=2,
                    help='Cosine Power Annealing Schedule Base, larger numbers make '
                         'the exponential more dominant, smaller make cosine more dominant, '
                         '1 returns to standard cosine annealing.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful for restarts)')
parser.add_argument('--warmup_epochs', type=int, default=5, help='num of warmup training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--mid_channels', type=int, default=32, help='C_mid channels in choke SharpSepConv')
parser.add_argument('--layers_of_cells', type=int, default=8, help='total number of cells in the whole network, default is 8 cells')
parser.add_argument('--layers_in_cells', type=int, default=4,
                    help='Total number of nodes in each cell, aka number of steps,'
                         ' default is 4 nodes, which implies 8 ops')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use cifar10 autoaugment https://arxiv.org/abs/1805.09501')
parser.add_argument('--random_eraser', action='store_true', default=False, help='use random eraser')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--no_architect', action='store_true', default=False, help='directly train genotype parameters, disable architect.')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--multi_channel', action='store_true', default=False, help='perform multi channel search, a completely separate search space')
parser.add_argument('--ops', type=str, default='OPS', help='which operations to use, options are OPS, DARTS_OPS and MULTICHANNELNET_OPS')
parser.add_argument('--primitives', type=str, default='PRIMITIVES',
                    help='which primitive layers to use inside a cell search space,'
                         ' options are PRIMITIVES, DARTS_PRIMITIVES AND MULTICHANNELNET_PRIMITIVES')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, metavar='PATH', default='',
                    help='evaluate model at specified path on training, test, and validation datasets')
parser.add_argument('--load', type=str, default='',  metavar='PATH', help='load weights at specified location')
parser.add_argument('--load_args', type=str, default='',  metavar='PATH',
                    help='load command line args from a json file, this will override '
                         'all currently set args except for --evaluate, and arguments '
                         'that did not exist when the json file was originally saved out.')
parser.add_argument('--weighting_algorithm', type=str, default='scalar',
                    help='which operations to use, options are '
                         '"max_w" (1. - max_w + w) * op, and scalar (w * op)')
# TODO(ahundt) remove final path and switch back to genotype
parser.add_argument('--final_path', type=str, default=None, help='path for final model')
parser.add_argument('--load_genotype', type=str, default=None, help='Name of genotype to be used')
args = parser.parse_args()



trainset = None
testset = None

def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()
        # print(f'data step time: {start-end}')
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        end=time.time()
        # print(f'model step time: {end-start}')


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy


def evaluate_model(model_cls):
    global trainset
    global testset

    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = args.batch_size
    trainset = CIFAR10(root='../data/cifar10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = CIFAR10(root='../data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    for epoch in range(args.epochs):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(accuracy)

    # report final test result
    nni.report_final_result(accuracy)
    


import os
from pathlib import Path


def evaluate_model_with_visualization(model_cls):
    model = model_cls()
    # dump the model into an onnx
    if 'NNI_OUTPUT_DIR' in os.environ:
        dummy_input = torch.zeros(1, 3, 32, 32)
        torch.onnx.export(model, (dummy_input, ),
                          Path(os.environ['NNI_OUTPUT_DIR']) / 'model.onnx')
    evaluate_model(model_cls)



model_space = Network(primitives=DARTS_PRIMITIVES, C=16, layers=args.layers_of_cells, steps=args.layers_in_cells)


from nni.retiarii.evaluator import FunctionalEvaluator
evaluator = FunctionalEvaluator(evaluate_model_with_visualization)

import nni.retiarii.strategy as strategy
search_strategy = strategy.TPE()



from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'cifar10_sharpsepconv'

exp_config.max_trial_number = 100   # spawn 4 trials at most
exp_config.trial_concurrency = 2  # will run two trials concurrently

exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True

exp.run(exp_config, 8081)

