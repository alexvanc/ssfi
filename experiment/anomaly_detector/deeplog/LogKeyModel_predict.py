import torch
import torch.nn as nn
import time
import argparse
import os

# Device configuration
device = torch.device("cpu")
# Hyperparameters
input_size = 1
num_classes = 269
model_path = 'model/Adam_batch_size=2048_epoch=5.pt'


def generate(tDict, file):
    with open(file) as f:
        templateID = []
        for line in f.readlines():
            tid = line.strip().split(',')[-1]
            if tid == 'EventId':
                continue
            templateID.append(tDict[tid])
    return templateID


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
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    TP = 0
    FP = 0
    num_anomaly = 0
    templateDict = {}
    with open('/home/FineLog/src/main/resources/models/nodemanager/templates.json') as fr:
        i = 0
        for line in fr.readlines():
            key = line.split(',')[0]
            if not key in templateDict:
                templateDict[key] = i
                i = i + 1
    # Test the model
    start_time = time.time()
    for root, dirs, files in os.walk('/home/FineLog/src/main/resources/data/hadoop/nodemanager/test/normal'):
        for f in files:
            line = generate(templateDict, os.path.join(root, f))
            with torch.no_grad():
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        FP += 1
                        break
    for root, dirs, files in os.walk('/home/FineLog/src/main/resources/data/hadoop/nodemanager/test/anomaly'):
        num_anomaly = len(files)
        for f in files:
            line = generate(templateDict, os.path.join(root, f))
            with torch.no_grad():
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        TP += 1
                        break
    # Compute precision, recall and F1-measure
    FN = num_anomaly - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
