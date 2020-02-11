import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import sys
import pymysql

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 1
num_epochs = 5
batch_size = 1024
model_dir = 'model'
train_size = 20


class Generator(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("10.0.0.49", "alex", "fake_passwd", "new_injection", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir

    def generate(self):
        sql="select MAX(id) from hadoop_log_template"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        template_space_size=results[0][0]+1
        trainDataset=self.generateDataset(template_space_size)
        return template_space_size,trainDataset
    
    def generateDataset(self,template_space_size):
        inputs=[]
        outputs=[]
        num_files = 0
        allJobs=os.listdir(self.dataDir)
        counter=0
        for fi in allJobs:
            if counter==train_size:
                break
            for dirpath,subdirs,files in os.walk(self.dataDir+"/"+fi):
                for tfile in files:
                    if tfile.endswith(".sequence2"):
                        with open(os.path.join(dirpath,tfile),'r') as f:
                            num_files=num_files+1
                            for line in f.readlines():
                                tmplSequence=[]
                                tmplStrSequence=line.strip().split(" ")
                                if len(tmplStrSequence)==1 and tmplStrSequence[0]=="":
                                    continue
                                for tmpStr in tmplStrSequence:
                                    x=int(tmpStr)
                                    if x<0:
                                        x=0
                                    if x>template_space_size:
                                        print("Error")
                                        print(os.path.join(dirpath,tfile))
                                    tmplSequence.append(x)
                                # tmplSequence=[int(x) for x in line.strip().split(" ")]
                                #make sure there is no negative value in tmplSequence
                            
                                length=len(tmplSequence)
                                #what if length<10
                                for i in range(length-window_size):
                                    inputs.append(tmplSequence[i:i+window_size])
                                    outputs.append(tmplSequence[i+window_size])
            counter=counter+1
        print('Number of files: {}'.format(num_files))
        print('Number of seqs: {}'.format(len(inputs)))
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
        return dataset


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
    parser.add_argument('-data_dir', default="/mnt/fi/hadoop-datasets/all_training", type=str)
    parser.add_argument('-model_dir', default="default", type=str)
    parser.add_argument('-train_size', default=20, type=int)
    parser.add_argument('-num_epochs', default=5, type=int)
    parser.add_argument('-batch_size', default=1024, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    train_size = args.train_size
    num_epochs= args.num_epochs
    batch_size=args.batch_size


    log = '{}_hidden_size={}_adam_batch_size={}_epoch={}'.format(args.model_dir, hidden_size, str(batch_size), str(num_epochs))

    generator=Generator(args.data_dir)
    model_dir=args.model_dir

    num_classes, seq_dataset = generator.generate()
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
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt'+str(train_size))
    print('Finished Training')

