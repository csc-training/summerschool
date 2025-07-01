import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import time

def get_model():
    return torchvision.models.resnet152(num_classes=100)

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='/scratch/project_462000956/data',
        train=True,
        download=True,
        transform=transform
    )
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=7)

    model = get_model()

    # Use DataParallel to wrap the model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    writer = SummaryWriter(log_dir="./logs/data_parallel_cifar100")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/data_parallel_cifar100/profiler"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(4):
            running_loss = 0.0
            start_group = time.time()
            start_epoch = time.time()

            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                torch.cuda.synchronize()

                prof.step()
                running_loss += loss.item()

                if i % 100 == 99:
                    end_group = time.time()
                    total_time = end_group - start_group
                    avg_iter_time = total_time / 100

                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, "
                          f"iteration_time: {total_time:.2f}s, "
                          f"time/iter: {avg_iter_time:.4f}s", 
                          flush=True)
                    
                    writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                    running_loss = 0.0
                    start_group = time.time()

            print(f"[{epoch + 1}], epoch time: {time.time()-start_epoch}s", flush=True)  # Reset timer for next group

    writer.close()

if __name__ == "__main__":
    train()
