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
import time
import uploader
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size=10
num_epochs=2
batch_size=4

class Generator(object):
    def __init__(self,dataDir):
        self.dataDir=dataDir

    def generate(self):
        trainDataset=self.generateDataset()
        return trainDataset
    
    def generateDataset(self,subdir):
        inputs=[]
        outputs=[]
        lengths=[]
        fault_ids=[]
        num_files = 0
        allJobs=os.listdir(self.dataDir+"/"+subdir)
        counter=0
        for fi in allJobs:
            length=0
            oneJobData=[]
            for dirpath,subdirs,files in os.walk(self.dataDir+"/"+subdir+"/"+fi):
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
                        if subdir=='normal':
                            outputs.append(0)
                        else:
                            outputs.append(1)
                        fault_ids.append(fi)
            counter=counter+1
        #normalization may be required
        # print('Number of files: {}'.format(num_files))
        # print('Number of seqs: {}'.format(len(inputs)))
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X_train = min_max_scaler.fit_transform(inputs)
        # dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(outputs))
        normalized_inputs=self.normalize(inputs,lengths)
        # print(len(normalized_inputs))
        return normalized_inputs,lengths,outputs, fault_ids
    
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
            scaler = preprocessing.StandardScaler().fit(tmp_dataset)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=1, type=int)
    parser.add_argument('-hidden_size', default=4, type=int)
    parser.add_argument('-data_dir', default="/data/hadoop-datasets/all_DEE", type=str)
    parser.add_argument('-model_dir', default="default", type=str)
    parser.add_argument('-train_size', default=50, type=int)
    parser.add_argument('-threshold', default=0.5, type=float)
    parser.add_argument('-num_epochs', default=0.5, type=int)
    parser.add_argument('-batch_size', default=5, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    train_size=args.train_size
    threshold=args.threshold
    data_dir=args.data_dir
    num_epochs=args.num_epochs
    batch_size=args.batch_size

    generator=Generator(data_dir)
    model_dir=args.model_dir
    num_classes=2

    #optional to upload data
    uploader=uploader.Uploader()
    # Test the model
    nx_data,nlengths,nlabels,ids=generator.generateDataset("normal")

    model=RNN(len(nx_data[0][0]),hidden_size,num_classes,num_layers).to(device)
    model_path='{}/{}_hidden_size={}_adam_batch_size={}_epoch={}.pt{}'.format(model_dir,model_dir, str(hidden_size), str(batch_size), str(num_epochs),str(train_size))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    TP = 0
    FP = 0
    TN=0
    FN=0
    num_anomaly = 0
    num_normal=0
    num_normal=len(nx_data)

    start_time = time.time()
    with torch.no_grad():
        for index in range(len(nx_data)):
            lineData=nx_data[index]
            length=nlengths[index]
            label=nlabels[index]
            seq = torch.tensor(lineData, dtype=torch.float).view(-1, len(lineData), len(lineData[0])).to(device)
            label = torch.tensor(label).view(-1).to(device)
            # print(torch.tensor(length,dtype=int))
            output = model(seq,torch.tensor([length]).long().to(device))
            sig_output=torch.sigmoid(output)
            if sig_output[0][0].item()<threshold:
                uploader.markMetrics(ids[index])
                FP=FP+1

    TN=num_normal-FP
    
        
     
    nx_data,nlengths,nlabels,ids=generator.generateDataset("anomaly")
    num_anomaly=len(nx_data)
    with torch.no_grad():
        for index in range(len(nx_data)):
            lineData=nx_data[index]
            length=nlengths[index]
            label=nlabels[index]
            seq = torch.tensor(lineData, dtype=torch.float).view(-1, len(lineData), len(lineData[0])).to(device)
            label = torch.tensor(label).view(-1).to(device)
            # print(torch.tensor(length,dtype=int))
            output = model(seq,torch.tensor([length]).long().to(device))
            sig_output=torch.sigmoid(output)
            if sig_output[0][0].item()<threshold:
                TP=TP+1
                uploader.markMetrics(ids[index])
    FN=num_anomaly-TP
    # print(TP)
    # print(FP)
    # print(num_anomaly)
    # Compute precision, recall and F1-measure
    # FN = num_anomaly - TP
    P=R=F1=ACC=FAR=-1
    try:
        P = 100 * TP / (TP + FP)
    except:
        pass

    try:
        R = 100 * TP / (TP + FN)
    except:
        pass

    try:
        F1 = 2 * P * R / (P + R)
    except:
        pass

    try:
        ACC = 100*(TP+TN) / (TP+TN+FP+FN)
    except:
        pass

    try:
        FAR = 100*(FP) / (FP+TN)
    except:
        pass

    print("data_dir: {}".format(data_dir))
    print("hidden_size: {}, num_epochs: {}, batch_size: {}, train_size: {}, threshold: {}".format(hidden_size,num_epochs,batch_size,train_size,threshold))
    print('false positive (FP): {}, false negative (FN): {}, true positive: {}, true negative: {}'.format(FP, FN, TP, TN))
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, False Alarm Rate: {:.3f}%, Accuracy: {:.3f}%'.format( P, R, F1, FAR, ACC))
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    print('Finished Predicting\n')

    uploader.upload(hidden_size,batch_size,num_epochs,train_size,threshold,data_dir,FP,FN,TP,TN,P,R,F1, \
        ACC,FAR)
    uploader.close()
    
