import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np

from mnist_dataload import get_datasets

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, ddp_model, device, train_loader, optimizer, epoch, rank):
    ddp_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx * dist.get_world_size() % args.log_interval == 0 and rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data) * dist.get_world_size()}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(ddp_model, device, test_loader, rank):
    ddp_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = ddp_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    tensor_test_loss = torch.full((32,), test_loss, device=device, dtype=torch.float32)
    dist.all_reduce(tensor_test_loss, op=dist.ReduceOp.SUM)
    test_loss = tensor_test_loss[0]

    tensor_correct = torch.full((32,), correct, device=device, dtype=torch.float32)
    dist.all_reduce(tensor_correct, op=dist.ReduceOp.SUM)
    correct = tensor_correct[0]

    if rank == 0:
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with DDP')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    dist.init_process_group("cpu:gloo,cuda:gloo")

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    dataset1, dataset2 = get_datasets('./data')
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(dataset2, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    train_kwargs = {
        'batch_size': args.batch_size,
        'sampler': train_sampler,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': True
    }
    
    test_kwargs = {
        'batch_size': args.test_batch_size,
        'sampler': test_sampler,
        'shuffle': False,
        'num_workers': 1,
        'pin_memory': True
    }
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Create model and wrap with DDP
    model = Net().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Create optimizer and scheduler
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        train(args, ddp_model, device, train_loader, optimizer, epoch, rank)
        test(ddp_model, device, test_loader, rank)
        dist.barrier()
        scheduler.step()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()