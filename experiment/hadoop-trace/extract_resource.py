# coding=utf-8
import sys
import os
import json
import re
import datetime as dt
from datetime import datetime

class ResourceParser(object):
    def __init__(self,resourceFilepath):
        self.resourceFilepath=resourceFilepath
        self.resourceDict={}
    
    def parse(self):
        if os.path.exists(self.resourceFilepath):
            resourceFile=open(self.resourceFilepath,'r')
            content=resourceFile.read()
            resourceFile.close()

            blocks=content.split("\n\n")
            for block in blocks:
                try:
                    self.parseBlock(block)
                except:
                    print(block)
            self.persistResource()
            
                
    def parseBlock(self,block):
        lines=block.strip().split("\n")
        timeText=lines[0]
        try:
            resourNames=lines[1]
        except:
            print(lines)
        resourceData=lines[2:]

        for resourceMatrix in resourceData:
            # columns=resourceMatrix.strip().split("\t")
            columns=re.split("\s{2,}",resourceMatrix)
            containerID=columns[0]
            containerName=columns[1]
            cpuUsageRatio=float(columns[2].split("%")[0])

            absMemUsageData=columns[3].split("/")
            memUsage=0.0
            memUsageQuota=0.0
            firstCell=absMemUsageData[0].strip()
            secondCell=absMemUsageData[1].strip()
            #memory is measured with MB
            if firstCell.endswith("KiB"):
                memUsage=float(firstCell.split("KiB")[0])/1000
            elif firstCell.endswith("MiB"):
                memUsage=float(firstCell.split("MiB")[0])
            elif firstCell.endswith("GiB"):
                memUsage=float(firstCell.split("GiB")[0])*1000
            elif firstCell.endswith("B"):
                memUsage=float(firstCell.split("B")[0])/1000000
            else:
                print("Unexpected absMemData1"+firstCell)

            if secondCell.endswith("KiB"):
                memUsageQuota=float(secondCell.split("KiB")[0])/1000
            elif secondCell.endswith("MiB"):
                memUsageQuota=float(secondCell.split("MiB")[0])
            elif secondCell.endswith("GiB"):
                memUsageQuota=float(secondCell.split("GiB")[0])*1000
            elif secondCell.endswith("B"):
                memUsageQuota=float(secondCell.split("B")[0])/1000
            else:
                print("Unexpected absMemData2"+secondCell)
            
            memUsageRatio=float(columns[4].split("%")[0])

            #network is measured with B
            networkInputUsage=0.0
            networkOutputusage=0.0
            networkData=columns[5].split("/")
            networkInput=networkData[0].strip()
            networkOutput=networkData[1].strip()
            
            if networkInput.endswith("kB"):
                networkInputUsage=float(networkInput.split("kB")[0])*1000
            elif networkInput.endswith("MB"):
                networkInputUsage=float(networkInput.split("MB")[0])*1000*1000
            elif networkInput.endswith("GB"):
                networkInputUsage=float(networkInput.split("GB")[0])*1000*1000*1000
            elif networkInput.endswith("B"):
                networkInputUsage=float(networkInput.split("B")[0])
            else:
                print("Unexpected networkData1"+networkInput)

            
            if networkOutput.endswith("kB"):
                networkOutputusage=float(networkOutput.split("kB")[0])*1000
            elif networkOutput.endswith("MB"):
                networkOutputusage=float(networkOutput.split("MB")[0])*1000*1000
            elif networkOutput.endswith("GB"):
                networkOutputusage=float(networkOutput.split("GB")[0])*1000*1000*1000
            elif networkOutput.endswith("B"):
                networkOutputusage=float(networkOutput.split("B")[0])
            else:
                print("Unexpected networkData1"+networkOutput)
            
            #block is measure with B
            blockInputUsage=0.0
            blockOutputUsage=0.0
            blockData=columns[6].split("/")
            blockInput=blockData[0].strip()
            blockOutput=blockData[1].strip()
            
            if blockInput.endswith("kB"):
                blockOutputUsage=float(blockInput.split("kB")[0])*1000
            elif blockInput.endswith("MB"):
                blockOutputUsage=float(blockInput.split("MB")[0])*1000*1000
            elif blockInput.endswith("GB"):
                blockOutputUsage=float(blockInput.split("GB")[0])*1000*1000*1000
            elif blockInput.endswith("B"):
                blockOutputUsage=float(blockInput.split("B")[0])
            else:
                print("Unexpected blockData1"+blockInput)

            
            if blockOutput.endswith("kB"):
                networkOutputusage=float(blockOutput.split("kB")[0])*1000
            elif blockOutput.endswith("MB"):
                networkOutputusage=float(blockOutput.split("MB")[0])*1000*1000
            elif blockOutput.endswith("GB"):
                networkOutputusage=float(blockOutput.split("GB")[0])*1000*1000*1000
            elif blockOutput.endswith("B"):
                networkOutputusage=float(blockOutput.split("B")[0])
            else:
                print("Unexpected blockData2"+blockOutput)
            
            processNumber=int(columns[7])

            start_time=dt.datetime.strptime(lines[0].strip(),"%Y-%m-%d_%H:%M:%S")

            resourcePoint={"container_name":containerName,"time":start_time.strftime("%Y-%m-%d %H:%M:%S"),"cpu_ratio":cpuUsageRatio,"mem_abs":memUsage,"mem_quota":memUsageQuota,"mem_ratio":memUsageRatio,"net_in":networkInputUsage,"net_out":networkOutputusage,"block_in":blockInputUsage,"block_out":blockOutputUsage,"process_num":processNumber}
            # resourcePoint={"container_name":containerName,"time":start_time,"cpu_ratio":cpuUsageRatio,"mem_abs":memUsage,"mem_quota":memUsageQuota,"mem_ratio":memUsageRatio,"net_in":networkInputUsage,"net_out":networkOutputusage,"block_in":blockInputUsage,"block_out":blockOutputUsage,"process_num":processNumber}

            if containerID in self.resourceDict:
                self.resourceDict[containerID].append(resourcePoint)
            else:
                self.resourceDict[containerID]=[resourcePoint]

    
    def persistResource(self):
        # because the monitoring data is already in time order,
        # we don't need to  sort the list again
        jsonFile=open("resource.dict","w")
        json.dump(self.resourceDict,jsonFile)
        jsonFile.close()
    

            


if __name__ == '__main__':
    if len(sys.argv)==2:
        parser=ResourceParser(sys.argv[1])
        parser.parse()
        

