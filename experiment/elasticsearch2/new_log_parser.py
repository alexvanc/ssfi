"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
import numpy as np
import pandas as pd
import hashlib
import datetime as dt
from datetime import datetime
from dateutil import parser as ps
import matplotlib.pyplot as plt
import time
from shutil import copyfile
import pickle 
import sys


class Logcluster:
    def __init__(self, logTemplate='', logIDL=None,sourceFile="empty",level="INFO"):
        self.logTemplate = logTemplate
        self.sourceFile=sourceFile
        self.level=level
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.timestep = None
        self.padding_dim = 10 # defult 10
        self.timeslice_dir = None
        self.padding_dir = None
        self.logData = None
        self.parsedDir = None
        self.testSize = None
        self.rootNode = Node()
        self.logCluL = []

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)

        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break
            
            if seqLen==1 and token in parentn.childD:
                #only for length==1
                retLogClust= Logcluster(seq,None,"empty","INFO")
                return retLogClust
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1
        
        

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            # If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    # seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)
    
    def loadData(self):
        if os.path.exists("/tmp/tree.dat"):
            treeDataFile=open("/tmp/tree.dat",'rb')
            self.rootNode=pickle.load(treeDataFile)
            treeDataFile.close()
        else:
            print("Cannot find the tree!!!")

        if os.path.exists("/tmp/clus.dat"):
            clusDataFile=open("/tmp/clus.dat",'rb')
            self.logCluL=pickle.load(clusDataFile)
            clusDataFile.close()
        else:
            print("Cannot find the Clus!!!")
    
    def saveData(self):
        updatedTreeDataFile=open("/tmp/tree.dat",'wb')
        pickle.dump(self.rootNode,updatedTreeDataFile)
        updatedTreeDataFile.close()

        updatedClusDataFile=open("/tmp/clus.dat",'wb')
        pickle.dump(self.logCluL,updatedClusDataFile)
        updatedClusDataFile.close()


    
    def new_parse(self,absoluteLogFilePath,log_format):
        # print('Parsing file: ' + absoluteLogFilePath)
        start_time = datetime.now()
        df_log=self.new_load_data(absoluteLogFilePath,log_format)
        count = 0
        for idx, line in df_log.iterrows():
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,_]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(self.rootNode, logmessageL)

            # Match no existing log cluster
            if matchCluster is None:
                # newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=None, sourceFile=absoluteLogFilePath,level=line['Level'])
                self.logCluL.append(newCluster)
                self.addSeqToPrefixTree(self.rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                # matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            # if count % 1000 == 0 or count == len(df_log):
            #     print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(df_log)))

        # print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        
    
    def new_load_data(self,absoluteLogFilePath,log_format):
        headers, regex = self.generate_logformat_regex(log_format)
        # pre-porcess a log file based on the pre-defined log-format
        df_log = self.log_to_dataframe(absoluteLogFilePath, regex, headers, log_format)
        return df_log
    
    def parse_logfile(self,parentFolder,filename,log_format):
        # print('Parsing file: ' + os.path.join(self.path, logName))
        structed_logfile=open(parentFolder+"/"+filename+"_structured.csv","w")
        structed_logfile.write("Tmpl_HASH,Level,Class,Tmpl\n")
        start_time = datetime.now()

        df_log=self.new_load_data(parentFolder+"/"+filename, log_format)


        count = 0
        for idx, line in df_log.iterrows():
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,_]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(self.rootNode, logmessageL)

            # Match no existing log cluster
            # this should be never executed
            if matchCluster is None:
                # print("Error Cannot match a cluster!!!")
                # print(parentFolder+"/"+filename)
                # print(logmessageL) 
                # sys.exit()               
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                self.logCluL.append(newCluster)
                self.addSeqToPrefixTree(self.rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                template_str=' '.join(matchCluster.logTemplate)
                template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
                structed_logfile.write(template_id+","+line['Level']+","+line.get("Class","log4j").replace("'",'"')+","+template_str+"\n")

        structed_logfile.close()
    
    def getAllCluster(self):
        
        allCluster=[]

        for lenList in self.rootNode.childD:
            parentn = self.rootNode.childD[lenList]
            for token in parentn.childD:
                if lenList==1:
                    clusData={}
                    clusData['length']=lenList
                    clusData['token']=token
                    clusData['level']='INFO'
                    clusData['tmpl']=token
                    clusData['source_file']='empty'
                    allCluster.append(clusData)
                    
                    # allCluster.append(Logcluster(logTemplate=[token],logIDL=None,sourceFile="empty",level="INFO"))
                else:
                    secondP=parentn.childD[token]
                    for cluster in secondP.childD:
                        clusData={}
                        clusData['length']=lenList
                        clusData['token']=token
                        clusData['level']=cluster.level
                        clusData['tmpl']=' '.join(cluster.logTemplate)
                        clusData['source_file']=cluster.sourceFile
                        allCluster.append(clusData)

                        # allCluster.append(cluster)
        return allCluster


        

        # currentDepth = 1
        # for token in seq:
        #     if currentDepth >= self.depth or currentDepth > seqLen:
        #         break

        #     if token in parentn.childD:
        #         parentn = parentn.childD[token]
        #     elif '<*>' in parentn.childD:
        #         parentn = parentn.childD['<*>']
        #     else:
        #         return retLogClust
        #     currentDepth += 1

        # logClustL = parentn.childD

        # retLogClust = self.fastMatch(logClustL, seq)








    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    def split_train_test(self, parsedDir, logData, testSize):
        self.parsedDir = parsedDir
        self.logData = logData
        self.testSize = testSize
        with open(self.parsedDir + '/' + self.logData + '_all_structured.csv', 'r+') as fall:
            all_lines = fall.readlines()
        train_lines, test_lines = [all_lines[0]], [all_lines[0]]
        train_lines += all_lines[1:-testSize]
        test_lines += all_lines[-testSize:]
        with open(self.parsedDir + '/' + self.logData + '_train_structured.csv', 'w+') as ftrain:
            for line in train_lines:
                ftrain.write(line)
        with open(self.parsedDir + '/' + self.logData + '_test_structured.csv', 'w+') as ftest:
            for line in test_lines:
                ftest.write(line)
        copyfile(self.parsedDir + '/' + self.logData + '_all_templates.csv', self.parsedDir + '/' + self.logData + '_train_templates.csv')
        copyfile(self.parsedDir + '/' + self.logData + '_all_templates.csv',self.parsedDir + '/' + self.logData + '_test_templates.csv')

    def cuttime(self, logName, timeslice_dir, padding_dir, label_dir, timestep, padding_dim, label):
        self.timeslice_dir = timeslice_dir
        self.padding_dir = padding_dir
        self.logName = logName
        self.timestep = timestep
        self.padding_dim = padding_dim
        self.label_dir = label_dir
        parsed_csv = pd.read_csv(self.savePath + '/' + self.logName + '_structured.csv')
        parsed_df = pd.DataFrame(parsed_csv)
        all_templates_dic = {}
        templates_csv = pd.read_csv(self.savePath + '/' + self.logName + '_templates.csv')
        i = 1
        for tem in templates_csv['EventId']:
            all_templates_dic[tem] = str(i)
            i += 1
        # print(all_templates_dic)
        start = ps.parse(parsed_df['Date'][1] + ' ' + parsed_df['Time'][1])
        end = ps.parse(parsed_df['Date'][len(parsed_df)-1]+' '+parsed_df['Time'][len(parsed_df)-1])
        length = int((end - start).total_seconds() / self.timestep)
        k = 0
        maxn = 0
        lens = []
        seqs = []
        lbs = []
        for i in range(0, length+1):
            s = start + i * dt.timedelta(seconds=self.timestep)
            e = start + (i + 1) * dt.timedelta(seconds=self.timestep)
            t = []
            lb = 0
            for j in range(k, len(parsed_df)):
                now = ps.parse(parsed_df['Date'][j]+' '+parsed_df['Time'][j])
                if s <= now < e:
                    t.append(all_templates_dic[parsed_df['EventId'][j]])
                    if parsed_df['Level'][j] == '[ERROR]':
                        lb = 1
                elif now >= e:
                    k = j
                    break
            if not t:
                t = ['0']
            maxn = max(maxn, len(t))
            seqs.append(t)
            lens.append(len(t))
            lbs.append(lb)
        with open(self.timeslice_dir + '/' + self.logName + '_timeslice.txt', 'w+') as f:
            for seq in seqs:
                f.write(' '.join(seq)+'\n')
        if label:
            with open(self.label_dir+'/'+self.logName+'_label.txt', 'w+') as f:
                for lb in lbs:
                    f.write(str(lb)+'\n')
        plt.hist(lens, cumulative=True, density=True, histtype='step', bins=1000)
        plt.ion() # not stop
        plt.savefig('pic/' + self.logName+'_distribution.png')
        print('maxn = ', maxn)
        # padding
        with open(self.timeslice_dir+'/'+self.logName+'_timeslice.txt', 'r+') as f:
            with open(self.padding_dir+'/'+self.logName+'_padding.txt', 'w+') as f1:
                for line in f.readlines():
                    line = line.strip().split()
                    if len(line) > self.padding_dim:
                        line = line[:self.padding_dim]
                    else:
                        line = line + ['0'] * (padding_dim - len(line))
                    f1.write(' '.join(line) + '\n')


# if __name__ == "__main__":
#     logdata = 'adc'
#     input_dir = 'data/{}/raw'.format(logdata)  # The input directory of log file
#     output_dir = 'data/{}/parsed'.format(logdata)  # The output directory of parsing results
#     timeslice_dir = 'data/{}/timeslice'.format(logdata)
#     padding_dir = 'data/{}/padding'.format(logdata)
#     label_dir = 'data/{}/label'.format(logdata)
#     log_format = '<Date> <Time> <Level> - <Component> - <Content>'
#     # Regular expression list for optional preprocessing (default: [])
#     regex = [
#         r'@[a-z0-9]+$',
#         r'[[a-z0-9\-\/]+]',
#         r'{.+}',
#         # r'blk_(|-)[0-9]+',  # block id
#         # r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
#         r'(\d+\.){3}\d+',
#         r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
#     ]
#     st = 0.5  # Similarity threshold
#     depth = 4  # Depth of all leaf nodes

#     parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
#     parser.parse('adc_all')

#     # split train and test
#     parser.split_train_test(parsedDir=output_dir, logData=logdata, testSize=183710)

#     # cuttime
#     timestep = 5
#     padding_dim = 170
#     parser.cuttime(logName='{}_train'.format(logdata), timeslice_dir=timeslice_dir, padding_dir=padding_dir, label_dir=label_dir, timestep=timestep, padding_dim=padding_dim, label=True)
#     parser.cuttime(logName='{}_test'.format(logdata), timeslice_dir=timeslice_dir, padding_dir=padding_dir, label_dir=label_dir, timestep=timestep, padding_dim=padding_dim, label=True)
if __name__ == "__main__":
    input_dir = '/tmp/logInput'  # The input directory of log file
    output_dir = '/tmp/logOutput'  # The output directory of parsing results
    log_format = '<UUID> <PID> <Date> <Time> <Level> <Class>: <Content>'
    log_format2 = '<UUID> <PID> <Date> <Time> <Level> <Unknown> <Class>: <Content>'
    # Regular expression list for optional preprocessing (default: [])
    regex = [
            r'@[a-z0-9]+$',
            r'[[a-z0-9\-\/]+]',
            r'{.+}',
            # r'blk_(|-)[0-9]+',  # block id
            # r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(\d+\.){3}\d+',
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
    ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
    parser.parse('test3.log')