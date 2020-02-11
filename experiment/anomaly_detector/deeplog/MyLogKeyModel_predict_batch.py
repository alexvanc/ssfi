import torch
import torch.nn as nn
import time
import argparse
import os
import pymysql
import sys
import uploader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 1
num_classes = 269
num_epochs = 5
batch_size = 4096
model_path = 'model/Adam_batch_size=2048_epoch=5.pt'

class Generator(object):
    def __init__(self,dataDir):
        self.db = pymysql.connect("127.0.0.1", "alex", "fake_passwd", "new_injection6", charset="utf8")
        self.cursor = self.db.cursor()
        self.dataDir=dataDir
    
    def getClassNumber(self):
        sql="select MAX(id) from hadoop_log_template"
        self.cursor.execute(sql)
        results=self.cursor.fetchall()
        return results[0][0]+1

    def generatePredictData(self,subdir,template_space_size):
        predictData=[]
        allJobs=os.listdir(self.dataDir+"/"+subdir)
        for fi in allJobs:
            oneJobData=[] #for all log files within one job
            oneJobData.append([]) #for input data
            oneJobData.append([]) #for output labels
            oneJobData.append(fi)
            for dirpath,subdirs,files in os.walk(self.dataDir+"/"+subdir+"/"+fi):
                for tfile in files:
                    if tfile.endswith(".sequence2"):
                        with open(os.path.join(dirpath,tfile),'r') as f:
                            for line in f.readlines():
                                #actually there should be only one line in one file
                                tmplSequence=[]#for one log file
                                tmplStrSequence=line.strip().split(" ")
                                if len(tmplStrSequence)==1 and tmplStrSequence[0]=="":
                                    continue
                                for tmpStr in tmplStrSequence:
                                    try:
                                        x=0
                                        if tmpStr=="None":
                                            x=0
                                        else:
                                            x=int(tmpStr)
                                        if x<0:
                                            x=0
                                        if x>template_space_size:
                                            print("Error")
                                            print(os.path.join(dirpath,tfile))
                                        tmplSequence.append(x)
                                    except:
                                        print("illegal sequence file: "+dirpath+"/"+tfile)
                                        sys.exit()

                                
                                # tmplSequence=[int(x) for x in line.strip().split(" ")]
                                #make sure there is no negative value in tmplSequence
                                for i in range(len(tmplSequence) - window_size):
                                    seq = tmplSequence[i:i + window_size]
                                    label = tmplSequence[i + window_size]
                                    oneJobData[0].append(seq)
                                    oneJobData[1].append(label)
            predictData.append(oneJobData)

        return predictData
    


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
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-data_dir', default="/mnt/fi/hadoop-datasets/all", type=str)
    parser.add_argument('-model_dir', default="default", type=str)
    parser.add_argument('-train_size', default=20, type=int)
    parser.add_argument('-num_epochs', default=5, type=int)
    parser.add_argument('-batch_size', default=1024, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    train_size=args.train_size
    num_epochs= args.num_epochs
    batch_size=args.batch_size
    data_dir=args.data_dir

    generator=Generator(data_dir)
    model_dir=args.model_dir
    num_classes=generator.getClassNumber()

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
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


    uploader=uploader.Uploader()
    # Test the model
    start_time = time.time()
    normalDataset=generator.generatePredictData("normal",num_classes)
    num_normal=len(normalDataset)
    for oneJobData in normalDataset:
        with torch.no_grad():
                seq = torch.tensor(oneJobData[0], dtype=torch.float).view(-1,window_size,input_size).to(device)
                label = torch.tensor(oneJobData[1],dtype=torch.long).to(device)
                output = model(seq)
                lineIndex=0
                predicted = torch.argsort(output, 1)
                for onePredict in predicted:
                    if label[lineIndex] not in onePredict[-num_candidates:]:
                        FP += 1
                        uploader.markDeep(oneJobData[2])
                        break
                    lineIndex=lineIndex+1

    TN=num_normal-FP

    anomalyDataset=generator.generatePredictData("anomaly",num_classes)
    num_anomaly=len(anomalyDataset)
    for oneJobData in anomalyDataset:
        with torch.no_grad():
                seq = torch.tensor(oneJobData[0], dtype=torch.float).view(-1,window_size,input_size).to(device)
                label = torch.tensor(oneJobData[1]).to(device)
                output = model(seq)
                lineIndex=0
                predicted = torch.argsort(output, 1)
                for onePredict in predicted:
                    if label[lineIndex] not in onePredict[-num_candidates:]:
                        TP += 1
                        uploader.markDeep(oneJobData[2])
                        break
                    lineIndex=lineIndex+1
    FN=num_anomaly-TP

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
    print("hidden_size: {}, num_epochs: {}, batch_size: {}, train_size: {}, num_candidates: {}".format(hidden_size,num_epochs,batch_size,train_size,num_candidates))
    print('false positive (FP): {}, false negative (FN): {}, true positive: {}, true negative: {}'.format(FP, FN, TP, TN))
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, False Alarm Rate: {:.3f}%, Accuracy: {:.3f}%'.format( P, R, F1, FAR, ACC))

    
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
    print('Finished Predicting')

     #optional to upload data
    uploader=uploader.Uploader()
    uploader.upload_deeplog(hidden_size,batch_size,num_epochs,train_size,num_candidates,data_dir,FP,FN,TP,TN,P,R,F1,ACC,FAR)
    uploader.close()
