package com.alex.ssfi.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class JobHelper {
	private final static Logger logger = LogManager.getLogger(JobHelper.class);

	public static void runJob(String ID, String fullClassName, String inputPath, String componentName, String jarName,
			String outputPath, long timeoutValue) {
		pack(ID, fullClassName, inputPath, componentName, jarName, outputPath);

		long runningTime = run(ID, fullClassName, inputPath, outputPath, timeoutValue);

		analyze(ID, outputPath, runningTime);

		clean(ID, fullClassName, inputPath, componentName, jarName, outputPath);
	}

	public static void runJob(String ID, String fullClassName, String inputPath, String outputPath, long timeoutValue) {

	}

	private static void pack(String ID, String fullClassName, String inputPath, String outputPath) {

	}

	private static void pack(String ID, String fullClassName, String inputPath, String componentName, String jarName,
			String outputPath) {
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("/bin/bash", "-c", outputPath + "/pack.sh");

		Map<String, String> env = pb.environment();
		env.put("ID", ID);
		env.put("INPUT", inputPath);
		env.put("OUTPUT", outputPath);
		env.put("COMPONENT", componentName);
		env.put("JARNAME", jarName);
		String target = fullClassName.replace('.', File.separatorChar) + ".class";
		env.put("TARGET", target);
		String className = target.substring(target.lastIndexOf(File.separatorChar) + 1);
		String classFolder = inputPath + File.separatorChar + componentName + File.separatorChar + jarName
				+ File.separatorChar + target.substring(0, target.lastIndexOf(File.separatorChar));
		env.put("FOLDER", classFolder);
		env.put("CLASS", className);

		try {
			Process p = pb.start();
			BufferedReader normalReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			StringBuilder normalBuilder = new StringBuilder();
			StringBuilder errorBuilder = new StringBuilder();

			String line;
			while ((line = normalReader.readLine()) != null) {
				normalBuilder.append(line + "\n");
			}

			while ((line = errorReader.readLine()) != null) {
				errorBuilder.append(line + "\n");
			}

			int exitVal = p.waitFor();
			if (exitVal == 0) {
				logger.info("Pack Success");
				logger.info(normalBuilder.toString());
				logger.info(errorBuilder.toString());
			} else {
				logger.warn("Pack Failed");
				logger.error(errorBuilder.toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());

		}
	}

	private static long run(String ID, String fullClassName, String inputPath, String outputPath, long timeoutValue) {
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("bash", "-c", outputPath + "/run.sh");
		Map<String, String> env = pb.environment();
//		pb.redirectOutput(new File("/tmp/runNormal.txt"));
//		pb.redirectError(new File("/tmp/runError.txt"));
		env.put("ID", ID);
		env.put("INPUT", inputPath);
		env.put("OUTPUT", outputPath);
		String target = fullClassName.replace(".", File.separator) + ".class";
		env.put("TARGET", target);
		String className = target.substring(target.lastIndexOf(File.separator));
		String classFolder = target.substring(0, target.lastIndexOf(File.separator));
		env.put("FOLDER", classFolder);
		env.put("CLASS", className);

		try {
			long startTime = System.currentTimeMillis();
			long nowTime = System.currentTimeMillis();
			Process p = pb.start();

			BufferedReader normalReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			StringBuilder normalBuilder = new StringBuilder();
			StringBuilder errorBuilder = new StringBuilder();

			while (p.isAlive() && (nowTime - startTime) < timeoutValue) {
				String line;
				if ((line = normalReader.readLine()) != null) {
					normalBuilder.append(line + "\n");
				}
				if ((line = errorReader.readLine()) != null) {
					errorBuilder.append(line + "\n");
				}
				Thread.sleep(100);
				nowTime = System.currentTimeMillis();
			}
			if ((nowTime - startTime) > timeoutValue) {
				// program hangs on
				kill(ID, outputPath);

			}
			int exitVal = p.waitFor();

			long endTime = System.currentTimeMillis();
			if (exitVal == 0) {

				logger.info("Run Success");
				logger.info(normalBuilder.toString());
				logger.info(errorBuilder.toString());

			} else {
				logger.info("Run Failed");
				logger.info(normalBuilder.toString());
				logger.info(errorBuilder.toString());

			}
			return endTime - startTime;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());
			return -1;
		}
	}

	private static void kill(String ID, String outputPath) {
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("bash", "-c", outputPath + "/kill.sh");
		Map<String, String> env = pb.environment();
		env.put("ID", ID);
		try {
			Process p = pb.start();
			BufferedReader normalReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			StringBuilder normalBuilder = new StringBuilder();
			StringBuilder errorBuilder = new StringBuilder();

			String line;
			while ((line = normalReader.readLine()) != null) {
				normalBuilder.append(line + "\n");
			}

			while ((line = errorReader.readLine()) != null) {
				errorBuilder.append(line + "\n");
			}
			int exitVal = p.waitFor();
			if (exitVal == 0) {
				logger.info("Kill Success");
				logger.info(normalBuilder.toString());
				logger.info(errorBuilder.toString());
			} else {
				logger.warn("Kill Failed");
				logger.info(normalBuilder.toString());
				logger.error(errorBuilder.toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());
		}

	}

	private static void clean(String ID, String fullClassName, String inputPath, String componentName, String jarName,
			String outputPath) {
		kill(ID, outputPath);
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("bash", "-c", outputPath + "/clean.sh");
		Map<String, String> env = pb.environment();
		env.put("ID", ID);
		env.put("INPUT", inputPath);
		env.put("OUTPUT", outputPath);
		env.put("COMPONENT", componentName);
		env.put("JARNAME", jarName);
		String target = fullClassName.replace('.', File.separatorChar) + ".class";
		env.put("TARGET", target);
		String className = target.substring(target.lastIndexOf(File.separatorChar) + 1);
		String classFolder = inputPath + File.separatorChar + componentName + File.separatorChar + jarName
				+ File.separatorChar + target.substring(0, target.lastIndexOf(File.separatorChar));
		env.put("FOLDER", classFolder);
		env.put("CLASS", className);
		try {
			Process p = pb.start();
			BufferedReader normalReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			StringBuilder normalBuilder = new StringBuilder();
			StringBuilder errorBuilder = new StringBuilder();

			String line;
			while ((line = normalReader.readLine()) != null) {
				normalBuilder.append(line + "\n");
			}

			while ((line = errorReader.readLine()) != null) {
				errorBuilder.append(line + "\n");
			}
			int exitVal = p.waitFor();
			if (exitVal == 0) {
				logger.info("Clean Success");
				logger.info(normalBuilder.toString());
				logger.error(errorBuilder.toString());
			} else {
				logger.warn("Clean Failed");
				logger.info(normalBuilder.toString());
				logger.error(errorBuilder.toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());
		}
	}

	private static void analyze(String ID, String outputPath, long runningTime) {
		ProcessBuilder pb = new ProcessBuilder();
		pb.command("bash", "-c", outputPath + "/analyze.sh");
		Map<String, String> env = pb.environment();
		env.put("ID", ID);
		env.put("OUTPUT", outputPath);
		env.put("ACT_FILE", outputPath + File.separatorChar + "activation.log");
		env.put("RTIME", "" + runningTime);
//		 env.remove("OTHERVAR");
//		 env.put("VAR2", env.get("VAR1") + "suffix");
//		 pb.directory(new File("myDir"));
		try {
			Process p = pb.start();
			BufferedReader normalReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			BufferedReader errorReader = new BufferedReader(new InputStreamReader(p.getErrorStream()));

			StringBuilder normalBuilder = new StringBuilder();
			StringBuilder errorBuilder = new StringBuilder();

			String line;
			while ((line = normalReader.readLine()) != null) {
				normalBuilder.append(line + "\n");
			}

			while ((line = errorReader.readLine()) != null) {
				errorBuilder.append(line + "\n");
			}
			int exitVal = p.waitFor();
			if (exitVal == 0) {
				logger.info("Analyze Success");
				logger.info(normalBuilder.toString());
				logger.error(errorBuilder.toString());
			} else {
				logger.warn("Analyze Failed");
				logger.info(normalBuilder.toString());
				logger.error(errorBuilder.toString());
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());
		}
	}

}
