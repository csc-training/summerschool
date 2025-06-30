import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
import time

# TODO: change the model to resnet50 and investigate the performance
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
    # TODO: change the batch size to 256 and investigate the results
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    writer = SummaryWriter(log_dir="./logs/single_gpu_cifar100")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/single_gpu_cifar100/profiler"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        start_epoch = time.time()
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

                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}, iteration_time: {total_time}s time/iter (100): {avg_iter_time:.4f}")

                    writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)

                    running_loss = 0.0
                    start_group = time.time()
            
            print(f"[{epoch + 1}], epoch time: {time.time()-start_epoch}s")  # Reset timer for next group
            
    writer.close()

if __name__ == "__main__":
    print("Training on single GPU")
    train()
