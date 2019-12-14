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
        pack2(ID, fullClassName, inputPath, outputPath);

        long runningTime = run(ID, fullClassName, inputPath, outputPath, timeoutValue);

        analyze(ID, outputPath, runningTime);

        clean2(ID, fullClassName, inputPath, outputPath);
    }

    private static void pack2(String ID, String fullClassName, String inputPath, String outputPath) {
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("/bin/bash", "-c", outputPath + "/pack.sh");

        Map<String, String> env = pb.environment();
        env.put("ID", ID);
        env.put("INPUT", inputPath);
        env.put("OUTPUT", outputPath);
        String target = fullClassName.replace('.', File.separatorChar) + ".class";
        env.put("TARGET", target);

        pb.redirectOutput(new File("/tmp/packNormal.txt"));
        pb.redirectError(new File("/tmp/packError.txt"));

        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 80000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 80000) {
                p.destroy();
                logger.error("Pack Fatal");
                forceKill(outputPath);
            }
            logger.info("Pack Success");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());

        }
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
        pb.redirectOutput(new File("/tmp/packNormal.txt"));
        pb.redirectError(new File("/tmp/packError.txt"));

        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 80000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 80000) {
                p.destroy();
                logger.error("Pack Fatal");
                forceKill(outputPath);
            }
            logger.info("Pack Success");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());

        }
    }

    private static long run(String ID, String fullClassName, String inputPath, String outputPath, long timeoutValue) {
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("bash", "-c", outputPath + "/run.sh");
        Map<String, String> env = pb.environment();
        pb.redirectOutput(new File("/tmp/runNormal.txt"));
        pb.redirectError(new File("/tmp/runError.txt"));
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

            while (p.isAlive() && (nowTime - startTime) < timeoutValue) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > timeoutValue) {
                // program hangs on
                p.destroy();
                kill(ID, outputPath);

            }

            long endTime = System.currentTimeMillis();
            logger.info("Run Success");

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
        pb.redirectOutput(new File("/tmp/killNormal.txt"));
        pb.redirectError(new File("/tmp/killError.txt"));
        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 80000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 80000) {
                // program hangs on
                p.destroy();
                forceKill(outputPath);

            }
            logger.info("Kill Success");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());
        }

    }

    private static void forceKill(String outputPath) {
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("/bin/bash", "-c", outputPath + "/forcekill.sh");

        Map<String, String> env = pb.environment();

        env.put("OUTPUT", outputPath);

        pb.redirectOutput(new File("/tmp/forceKillNormal.txt"));
        pb.redirectError(new File("/tmp/forceKillError.txt"));

        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            // pack process cannot be longer that one minute
            while (p.isAlive() && (nowTime - startTime) < 60000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }

            logger.info("ForceKill Success");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());

        }
    }

    private static void clean2(String ID, String fullClassName, String inputPath, String outputPath) {
        kill(ID, outputPath);
        ProcessBuilder pb = new ProcessBuilder();
        pb.command("bash", "-c", outputPath + "/clean.sh");
        Map<String, String> env = pb.environment();
        env.put("ID", ID);
        env.put("INPUT", inputPath);
        env.put("OUTPUT", outputPath);
        String target = fullClassName.replace('.', File.separatorChar) + ".class";
        env.put("TARGET", target);
        pb.redirectOutput(new File("/tmp/cleanNormal.txt"));
        pb.redirectError(new File("/tmp/cleanError.txt"));
        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 130000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 130000) {
                // program hangs on
                p.destroy();
                forceKill(outputPath);
                logger.error("cleaning hang error");
            }

            logger.info("Clean Success");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());
        }
    }

    private static void clean(String ID, String fullClassName, String inputPath, String componentName, String jarName,
            String outputPath) {
    	forceKill(outputPath);
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
        pb.redirectOutput(new File("/tmp/cleanNormal.txt"));
        pb.redirectError(new File("/tmp/cleanError.txt"));
        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 130000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 130000) {
                // program hangs on
                p.destroy();
                forceKill(outputPath);
                logger.error("cleaning hang error");
            }

            logger.info("Clean Success");

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
        pb.redirectOutput(new File("/tmp/analyzeNormal.txt"));
        pb.redirectError(new File("/tmp/analyzeError.txt"));
        // env.remove("OTHERVAR");
        // env.put("VAR2", env.get("VAR1") + "suffix");
        // pb.directory(new File("myDir"));
        try {
            long startTime = System.currentTimeMillis();
            long nowTime = System.currentTimeMillis();
            Process p = pb.start();
            while (p.isAlive() && (nowTime - startTime) < 60000) {
                Thread.sleep(100);
                nowTime = System.currentTimeMillis();
            }
            if ((nowTime - startTime) > 60000) {
                // program hangs on
                p.destroy();
                logger.error("Analyze Fatal");
            }

            logger.info("Analyze Success");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            logger.error(e.getMessage());
        }
    }

}
