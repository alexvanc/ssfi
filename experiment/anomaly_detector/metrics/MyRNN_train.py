import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import json
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import os
import numpy as np
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size=10
num_epochs=2
batch_size=4

class Generator(object):
    def __init__(self,dataDir):
        self.dataDir=dataDir

    
    def generateDataset(self):
        inputs=[]
        outputs=[]
        lengths=[]
        num_files = 0
        allJobs=os.listdir(self.dataDir)
        counter=0
        for fi in allJobs:
            if counter==train_size:
                break
            length=0
            oneJobData=[]
            for dirpath,subdirs,files in os.walk(self.dataDir+"/"+fi):
                for tfile in files:
                    if tfile=="resource.dat":
                        num_files=num_files+1
                        with open(os.path.join(dirpath,tfile),'r') as f:
                            dataList=json.load(f)
                            for datacell in dataList:
                                oneData=[]
                                for key,value in datacell.items():
                                    if key!="time" and key!='container_name':
                                        oneData.append(value)
                                oneJobData.append(oneData)
                                length=length+1
            inputs.append(oneJobData)
            lengths.append(length)
            outputs.append(0)
            counter=counter+1
        #normalization may be required
        print('Number of files: {}'.format(num_files))
        print('Number of seqs: {}'.format(len(inputs)))
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X_train = min_max_scaler.fit_transform(inputs)
        # dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(outputs))
        normalized_inputs=self.normalize(inputs,lengths)
        print(len(normalized_inputs))
        return normalized_inputs,lengths,outputs

    
    #normalization without padding
    def normalize(self, xdataset,lengths):
        tmp_dataset=[]
        for line in xdataset:
            for cell in line:
                tmp_dataset.append(cell)
        # tmp_dataset=np.array(tmp_dataset)
        scaler=None
        if os.path.exists("normalization.dat"):
            with open("normalization.dat",'rb') as nfile:
                scaler=pickle.load(nfile)
        else:
            print("New normalization data!")
            scaler = preprocessing.StandardScaler().fit(tmp_dataset)
            with open("normalization.dat",'wb') as nfile:
                pickle.dump(scaler,nfile)
        tmp_dataset2=scaler.transform(tmp_dataset)
        # print(tmp_dataset2.shape)
        result_dataset=[]
        current_index=0
        for line_index in range(len(lengths)):
            result_dataset.append(tmp_dataset2[current_index:current_index+lengths[line_index]].tolist())
            current_index=current_index+lengths[line_index]
        return result_dataset


        

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)


    def forward(self, input, length):    
        packed = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False).float()
        pack_output, hidden = self.gru(packed)
        
        gru_output, _ = pad_packed_sequence(pack_output, batch_first=True)
        output = self.decoder(hidden.view(-1, self.hidden_size))
        return output


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-data_dir', default="/data/hadoop-datasets/all_training", type=str)
    argparser.add_argument('-model_dir', default="default", type=str)
    argparser.add_argument('--hidden_size', type=int, default=4)
    argparser.add_argument('--n_layers', type=int, default=1)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--chunk_len', type=int, default=200)
    argparser.add_argument('-train_size', default=20, type=int)
    argparser.add_argument('-num_epochs', default=2, type=int)
    argparser.add_argument('-batch_size', default=5, type=int)
    
    args = argparser.parse_args()

    num_layers = args.n_layers
    hidden_size = args.hidden_size
    train_size = args.train_size
    model_dir=args.model_dir
    data_dir=args.data_dir
    num_epochs=args.num_epochs
    batch_size=args.batch_size

    
    generator=Generator(data_dir)
    x_data,lengths,labels=generator.generateDataset()

    
    num_classes=2

    model=RNN(len(x_data[0][0]),hidden_size,num_classes,num_layers)
    print(len(x_data[0][0]))
     # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    #Train the model
    total_step = int(len(x_data)/batch_size)
    print('total_step: {}'.format(total_step))
    log = '{}_hidden_size={}_adam_batch_size={}_epoch={}'.format(model_dir, str(hidden_size),str(batch_size), str(num_epochs))

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0.0
        for step in range(total_step):
            x_tdate=[]
            tlengths=[]
            tlabels=[]

            if step==total_step:
                x_tdata=x_data[step*batch_size:]
                tlengths=lengths[step*batch_size:]
                tlabels=labels[step*batch_size:]
            else:
                x_tdata=x_data[step*batch_size:(step+1)*batch_size]
                tlengths=lengths[step*batch_size:(step+1)*batch_size]
                tlabels=labels[step*batch_size:(step+1)*batch_size]
            optimizer.zero_grad()


            #padding the batch data
            padding_length=max(tlengths)
            for index in range(len(x_tdata)):
                left_length=padding_length-len(x_tdata[index])
                if left_length>0:
                    for index2 in range(left_length):
                        # print(x_tdata[index])
                        # np.append(x_tdata[index],np.zeros(9),axis=0)
                        x_tdata[index].append([0,0,0,0,0,0,0,0,0])
            # print(len(x_tdata[0]))
            # print(tlengths)
            output=model(torch.tensor(x_tdata,dtype=float),torch.tensor(tlengths,dtype=float))
            # torch.argmax(output,dim=1)
            # loss = criterion(torch.argmax(output,dim=1), torch.tensor(tlabels).long())
            loss = criterion(output, torch.tensor(tlabels).long())

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # print statistics
            if step % 5 == 4:    # print every 20 mini-batches
                print('Epoch: [{}/{}], Iter: [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, num_epochs, step + 1, total_step, train_loss / 5))
                train_loss = 0
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt'+str(train_size))
    print('Finished Training')


