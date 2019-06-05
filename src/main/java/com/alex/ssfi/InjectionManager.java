package com.alex.ssfi;

import java.util.List;
import java.util.Random;

import com.alex.ssfi.util.BatchRun;
import com.alex.ssfi.util.Configuration;
import com.alex.ssfi.util.SingleRun;

public class InjectionManager {
	
	private static InjectionManager instance = new InjectionManager();

	private InjectionManager() {
	}

	public static InjectionManager getManager(){
		return instance;
	}
	
	public void peformInjection(Configuration config) {
		if (config.getInjectionMode().equals("batch")){
			this.performBatchInjection(config.getBatchRun(),config.getInputPath(), config.getOutputPath());
		}else {
			this.performSingleInjection(config.getSingleRun(), config.getInputPath(), config.getOutputPath());
		}
	}
	
	private void performSingleInjection(SingleRun singleInjection, String input, String ouput) {
		
		
	}
	
	private void performBatchInjection(BatchRun batchInjection,String input, String ouput) {
		List<SingleRun> allRuns=batchInjection.getFaultList();
		String distributionMode=batchInjection.getDistributionMode();
		if((distributionMode==null) || distributionMode.equals("random")) {//random distribution mode
			int[] injectionNumber=this.randomDistribute(batchInjection.getCounter(), allRuns.size());
			for(int i=0;i<allRuns.size();i++) {
				for(int j=0;j<injectionNumber[i];i++) {
					this.performSingleInjection(allRuns.get(i), input, ouput);
				}
			}
		}else {// in the fixed distribution mode
			for (int i=0;i<allRuns.size();i++) {
				SingleRun singleRun=allRuns.get(i);
				int injectionNumber=(int) (batchInjection.getCounter()*singleRun.getDistribution());
				for (int j=0;j<injectionNumber;i++) {
					this.performSingleInjection(singleRun, input, ouput);
				}
			}
		}
		generateReport();
	}
	private int[] randomDistribute(int total,int typeNumber) {
		int[] distribution=new int[typeNumber];
		Random rand=new Random();
		int seed=100;
		for(int i=0;i<(typeNumber-1);i++) {
			int tmp=rand.nextInt(seed)+1;
			distribution[i]= (int) (total*((double)seed/(double)100));
			seed-=tmp;
		}
		distribution[typeNumber-1]=(int) (total*((double)seed/(double)100));
		return distribution;
	}
	
	/*
	 * Summarize the fault injection experiments
	 */
	private void generateReport() {
		//TODO
	}
		


}
