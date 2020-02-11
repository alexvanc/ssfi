import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 1
num_epochs = 5
batch_size = 4096
model_dir = 'model'
log_type = 'resourcemanager'
log = '{}_adam_batch_size={}_epoch={}'.format(log_type, str(batch_size), str(num_epochs))

def generate():
    inputs = []
    outputs = []
    num_files = 0
    templateDict = {}
    with open('/home/FineLog/src/main/resources/models/' + log_type + '/templates.json') as fr:
        num_classes = 0
        for line in fr.readlines():
            key = line.split(',')[0]
            if not key in templateDict:
                templateDict[key] = num_classes
                num_classes = num_classes + 1
    for root, dirs, files in os.walk('/home/FineLog/src/main/resources/data/hadoop/' + log_type + '/train'):
        for f in files:
            num_files += 1
            with open(os.path.join(root, f)) as fr:
                templateID = []
                for line in fr.readlines():
                    tid = line.strip().split(',')[-1]
                    if tid == 'EventId':
                        continue
                    templateID.append(templateDict[tid])
            for i in range(len(templateID) - window_size):
                inputs.append(templateID[i:i + window_size])
                outputs.append(templateID[i + window_size])
    print('Number of files: {}'.format(num_files))
    print('Number of seqs: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return num_classes, dataset



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    num_classes, seq_dataset = generate()
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    total_step = len(dataloader)
    print('total_step: {}'.format(total_step))
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # print statistics
            if step % 50 == 49:    # print every 100 mini-batches
                print('Epoch: [{}/{}], Iter: [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, num_epochs, step + 1, total_step, train_loss / 50))
                train_loss = 0
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    print('Finished Training')

