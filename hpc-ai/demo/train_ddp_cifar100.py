import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import argparse
import time

def get_model():
    return torchvision.models.resnet152(num_classes=100)  # CIFAR-100 has 100 classes

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match ResNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100 stats
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='/scratch/project_462000956/data',
        train=True,
        download=True,
        transform=transform
    )

    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = DataLoader(trainset, batch_size=int(128/world_size), sampler=sampler, num_workers=7, pin_memory=True)

    model = get_model().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    writer = SummaryWriter(log_dir=f"./logs/ddp_cifar100/rank_{rank}")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./logs/ddp_cifar100/profiler_rank_{rank}"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(4):
            dist.barrier()  # Sync all ranks before timing
            start_epoch = time.time()
            sampler.set_epoch(epoch)
            running_loss = 0.0
            iter_count = 0
            start_group = time.time()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()

                prof.step()
                running_loss += loss.item()

                iter_count += 1

                if i % 100 == 99:
                    end_group = time.time()
                    total_group_time = end_group - start_group
                    avg_iter_time = total_group_time / 100

                    print(f"[{epoch + 1}, {i + 1}] rank: {rank}, loss: {running_loss / 100:.3f}, iteration_time: {total_group_time}s time/iter (100): {avg_iter_time:.4f}")

                    writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                    running_loss = 0.0
                    start_group = time.time()  # Reset timer for next 10 iterations
            
            dist.barrier()  # Sync all ranks after epoch
            end_epoch = time.time()
            
            epoch_time = end_epoch - start_epoch
            if rank == 0:
                print(f"[{epoch + 1}], epoch time: {time.time()-start_epoch}s")  # Reset timer for next group

    writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=int(os.environ["RANK"]))
    parser.add_argument("--world_size", type=int, default=int(os.environ["WORLD_SIZE"]))
    args = parser.parse_args()
    if args.rank == 0:
        print(f"Running DDP with {args.world_size} GPUs")
    train(args.rank, args.world_size)
