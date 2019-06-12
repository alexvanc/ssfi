package com.alex.ssfi.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;

public class JobHelper {
//	private final Logger logger=LogManager.getLogger("result_recorder");
	public static void run(String ID,String fullClassName,String inputPath,String outputPath) {
		int timeLimitation=4000;
		long start=System.currentTimeMillis();
		int timeElapsed=0;
		Task task=new Task(ID, fullClassName, inputPath, outputPath);
		task.start();
		while(task.isAlive()) {
			long now=System.currentTimeMillis();
			timeElapsed+=now-start;
			if(timeElapsed>timeLimitation) {
				//timeout
				try {
					FileWriter writer=new FileWriter("/tmp/halo_result.txt");
					writer.write("hang");
					writer.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					System.out.println("Fatal! Cannot write running result");
				}finally{
					task.stop();
				}
			}
		}
		
	}

}

class Task extends Thread{
	private String ID;
	private String fullClassName;
	private String inputPath;
	private String outputPath;
	public Task(String ID,String fullClassName,String inputPath,String outputPath) {
		this.ID=ID;
		this.fullClassName=fullClassName;
		this.inputPath=inputPath;
		this.outputPath=outputPath;
		
	}
	@Override
	public void run() {
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("bash","-c","/home/alex/work/soot/ssfi/target/classes/job.sh");
		 Map<String, String> env = pb.environment();
		 env.put("ID",this.ID);
		 env.put("INPUT", this.inputPath);
		 env.put("OUTPUT", this.outputPath);
		 String target=this.fullClassName.replace(".", File.separator)+".class";
		 env.put("TARGET",target );
		 String className=target.substring(target.lastIndexOf(File.separator));
		 String classFolder=target.substring(0, target.lastIndexOf(File.separator));
		 env.put("FOLDER", classFolder);
		 env.put("CLASS",className);
//		 env.remove("OTHERVAR");
//		 env.put("VAR2", env.get("VAR1") + "suffix");
//		 pb.directory(new File("myDir"));
		 try {
			Process p = pb.start();
			BufferedReader reader=new BufferedReader(new InputStreamReader(p.getInputStream()));
			StringBuilder builder=new StringBuilder();
			String line;
			int lineCounter=0;
			while((line=reader.readLine())!=null) {
				lineCounter++;
				builder.append(line+"\n");
			}
			int exitVal=p.waitFor();
			if(exitVal==0) {
				System.out.println("Success");
				System.out.println(builder.toString());
			}else {
				System.out.println("Failed");
				System.out.println(builder.toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println(e.getMessage());
		}
	}
}
