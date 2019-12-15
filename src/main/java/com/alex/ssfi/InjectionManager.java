package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.BatchRun;
import com.alex.ssfi.util.Configuration;
import com.alex.ssfi.util.RunningParameter;
import com.alex.ssfi.util.SingleRun;

import soot.BodyTransformer;
import soot.G;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.Transform;

public class InjectionManager {
	private static final Logger logger=LogManager.getLogger(InjectionManager.class);
//	private static final Logger recorder=LogManager.getLogger("inject_recorder");
	
	private static InjectionManager instance = new InjectionManager();

	private InjectionManager() {
	}

	public static InjectionManager getManager(){
		return instance;
	}
	
	public void peformInjection(Configuration config) {
		if (config.getInjectionMode().equals("batch")){
			logger.info("In Batch mode");
			this.performBatchInjection(config.getBatchRun(),config.getInputPath(), config.getOutputPath());
		}else {
			logger.info("In Single mode");
			this.performSingleInjection(config.getSingleRun(), config.getInputPath(), config.getOutputPath());
		}
	}
	
	private void performSingleInjection(SingleRun singleInjection, String input, String ouput) {
		
		List<String> allClassName;
		allClassName=this.getFullClassName(input,singleInjection.getPackagePattern(),singleInjection.getClassPattern());
		
		if ((allClassName==null)||allClassName.size()==0) {
			logger.error("Failed to find a posssible targeting class file!");
			//generate unqualified report
			return;
		}
		logger.debug("Prepare to inject one fault with possible classes number"+ allClassName.size());
		while(true) {
			int length=allClassName.size();
			if(length==0) {
				//generate unqualified report
				logger.debug("Failed to inject one fault");
				return;
			}
			int index=new Random().nextInt(length);
			 
			String targetPath=allClassName.get(index);
			allClassName.remove(index);
			int start=targetPath.indexOf(input)+input.length()+1;
			int end=targetPath.indexOf(".class");
			String classWithPackge=targetPath.substring(start,end).replace(File.separatorChar, '.');
//			System.out.println(start+" "+classWithPackge);
			if(this.injectFault(singleInjection, classWithPackge, input, ouput)) {
				logger.debug("Succeed to inject one fault");
				break;
			}
		}

	}
	private boolean injectFault(SingleRun singleInjection,String classWithPackage,String input, String output) {
		Scene.v().addBasicClass("java.io.FileWriter",SootClass.SIGNATURES);
		Scene.v().addBasicClass("java.io.Writer",SootClass.SIGNATURES);
		Scene.v().addBasicClass("java.io.OutputStreamWriter",SootClass.SIGNATURES);
		RunningParameter parameter=new RunningParameter(singleInjection, output);
		logger.debug("Start to inject: "+parameter.getID()+" with "+singleInjection.getType()+" into "+classWithPackage);
		BodyTransformer transformer=TransformerHelper.getTransformer(singleInjection.getType(),parameter);
    	PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", transformer) );
    	
    	try {
    		MainWrapper.main(this.buildArgs(classWithPackage,input,output));
    		if(parameter.isInjected()) {
    			logger.debug("Succeed to inject:"+parameter.getID()+" with "+singleInjection.getType()+" into "+classWithPackage);
    			//Execute a job to observe the FI results
    			//JobHelper.run(parameter.getID(),classWithPackage,input,output);
    			return true;
    		}else {
    			logger.debug("Failed to inject:"+parameter.getID()+" with "+singleInjection.getType()+" into "+classWithPackage);
    			return false;
    		}
    		
    	}catch(Exception e) {
    		logger.debug("Failed to inject2:"+parameter.getID()+" with "+singleInjection.getType()+" into "+classWithPackage);
    		logger.error(e.getMessage());
    		return false;
    	}finally{
    		PackManager.v().getPack("jtp").remove("jtp.instrumenter");
    		G.reset();
    	}
    }
	
	private String[] buildArgs(String classWithPackage,String input,String output) {
		//TODO should guarantee input is the classPath 
		String[]args=new String[9];
		args[0]="-cp";
		args[1]=".:"+input;
		args[2]="-pp";
		args[3]="-p";
		args[4]="jb";
		args[5]="use-original-names:true";
		// args[6]="-f";
		// args[7]="jimple";
		// args[8]="-d";
		// args[9]=output;
		// args[10]=classWithPackage;
       args[6]="-d";
       args[7]=output;
       args[8]=classWithPackage;
		return args;	
	}
	
	
	private List<String> getFullClassName(String input,String subPackage, String className){
//		List<String> allFullClassName=new ArrayList<String>();
		String rootPackage=null;
		if((subPackage==null)||(subPackage=="")){
			rootPackage=input;
		}else {
			rootPackage=input+File.separator+subPackage.replace(".", File.separator);
		}
		
		//Check whether we can directly use apache with regex expression later
//		Collection files = FileUtils.listFiles(
//				  dir, 
//				  new RegexFileFilter("^(.*?)"), 
//				  DirectoryFileFilter.DIRECTORY
//				);
		File rootPack=new File(rootPackage);
		if((!rootPack.exists())||(!rootPack.isDirectory())) {
			return null;
		}
		return this.getQualifiedFullClass(rootPack,className);
		
	}
	private List<String> getQualifiedFullClass(File folder,String className){
		List<String> allFullClassName=new ArrayList<String>();
		File[] files=folder.listFiles();
		for(int i=0;i<files.length;i++) {
			File tmpFile=files[i];
//			if ((!tmpFile.isDirectory())&&(tmpFile.getAbsolutePath().endsWith(className+".class")) {
//				allFullClassName.add(tmpF)
//			}
			if(!tmpFile.isDirectory()) {
				if (className==null) {
					className="";
				}
				if(tmpFile.getAbsolutePath().endsWith(className+".class")) {
					allFullClassName.add(tmpFile.getAbsolutePath());
				}
			}else {
				allFullClassName.addAll(this.getQualifiedFullClass(tmpFile, className));
			}
		}
		return allFullClassName;
		
	}
	
	private void performBatchInjection(BatchRun batchInjection,String input, String ouput) {
		List<SingleRun> allRuns=batchInjection.getFaultList();
		
		String distributionMode=batchInjection.getDistributionMode();
		logger.info("Batch size: "+batchInjection.getCounter());
		if((distributionMode==null) || distributionMode.equals("random")) {//random distribution mode
			int[] injectionNumber=this.randomDistribute(batchInjection.getCounter(), allRuns.size());
			logger.debug("Random distribution: "+injectionNumber.toString());
			for(int i=0;i<allRuns.size();i++) {
				for(int j=0;j<injectionNumber[i];i++) {
					this.performSingleInjection(allRuns.get(i), input, ouput);
				}
			}
		}else {// in the fixed distribution mode
			for (int i=0;i<allRuns.size();i++) {
				SingleRun singleRun=allRuns.get(i);
				int injectionNumber=(int) (batchInjection.getCounter()*singleRun.getDistribution());
				for (int j=0;j<injectionNumber;j++) {
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
