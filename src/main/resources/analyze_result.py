# coding=utf-8
import sys
import os

class Analyzer(object):
    def __init__(self,filename):
        self.ID=os.environ['ID']
        self.fileName=filename
    
    def analyze(self):
        resultFile=open("/tmp/result.log",'a+')
        if os.path.exists(filename):
            haloOutputFile=open(self.fileName,'r')
            content=haloOutputFile.read()
            if(content=="300"):
                resultFile.write(self.ID+"\t"+"Benign")
            elif(content=="hang":
                resultFile.write(self.ID+"\t"+"Hang")
            else:
                resultFile.write(self.ID+"\t"+"SDC")
            haloOutputFile.close())
        else:
            resultFile.write(self.ID+"\t"+"Crash")
        
        resultFile.close()

            


if __name__ == '__main__':
    if len(sys.argv)==2:
        analyzer=Analyzer(sys.argv[1])
        analyzer.analyze()
        

