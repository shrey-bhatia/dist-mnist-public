import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Timing utilities
class Timer:
    def __init__(self, name=None):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            print(f'{self.name} took {self.interval:.3f} seconds')

# Define the model - same as distributed version
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(epochs=5, batch_size=64):
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create model
    model = ConvNet().to(device)
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track performance metrics
    epoch_times = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        batches_processed = 0
        
        # Time the epoch
        with Timer(f"Epoch {epoch+1}"):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                batches_processed += 1
                
                if batch_idx % 100 == 0:
                    avg_loss = running_loss / batches_processed
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {avg_loss:.4f}')
        
        # Compute average loss for the epoch
        avg_epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} complete, Avg Loss: {avg_epoch_loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'single_node_mnist_model.pth')
    print(f"Model saved to single_node_mnist_model.pth")

if __name__ == "__main__":
    # Train the model
    train()