package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.Configuration;
import com.alex.ssfi.util.RunningParameter;

import soot.BodyTransformer;
import soot.G;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.Transform;

public class InjectionManager {
    private static final Logger logger = LogManager.getLogger(InjectionManager.class);

    private static InjectionManager instance;

    private final boolean debugMode;
    private final String inputPath;
    private final String outputPath;
    private final String dependencyPath;
    private final String subPackage;
    private String className;
    private final RunningParameter runningParameters;

    private InjectionManager(Configuration config) { // 根据config构建InjectionManage实例
        this.debugMode = config.isDebug(); // 获取是否开启debug模式的标识
        this.inputPath = config.getInputPath();  // 获取输入路径
        this.outputPath = config.getOutputPath();  // 获取输出路径
        this.dependencyPath = config.getDependencyPath(); // 获取依赖路径
        this.subPackage = config.getPackagePattern(); // 获取目标包名
        this.className = config.getClassPattern(); // 获取目标类名
        this.runningParameters = new RunningParameter(config); // 根据config构建RunningParameter实例
    }

    public static InjectionManager getManager(Configuration config) { // 获取InjectionManage实例
        if (instance == null) {
            instance = new InjectionManager(config);
        }
        return instance;
    }

    // 根据配置，执行故障注入
    public boolean startInjection() {
        List<String> allClassName = this.getFullClassName(); // 获取所有class的文件路径

        if (allClassName == null || allClassName.isEmpty()) { // 若没有待故障注入的class, 则报错
            logger.error("startInjection: Failed to find a possible targeting class file!");
            return false;
        }
        logger.debug("Prepare to inject one fault with possible classes number" + allClassName.size());
        while (true) { // 一直尝试进行故障注入，直到成功注入故障一次或对所有class进行过尝试
            if (allClassName.isEmpty()) { // 若待注入故障的class集为空，则报错
                // generate unqualified report
                logger.error("Failed to find a qualified class to inject a fault!");
                return false;
            }

            int index = new Random().nextInt(allClassName.size()); // 随机选取待注入故障的class的索引
            String targetPath = allClassName.remove(index); // 获取待注入故障的class，并从待注入故障的class集中删除将要尝试的class

            int start = targetPath.indexOf(inputPath) + inputPath.length() + 1; // 计算目标类名，在文件路径中的开始位置
            int end = targetPath.indexOf(".class"); // 计算目标类名，在文件路径中的结束位置
            String classWithPackage = targetPath.substring(start, end).replace(File.separatorChar, '.'); // 计算class的名称（带包名）


            if (this.injectFault(classWithPackage)) {
                logger.info("Succeed to inject one fault");
                return true;
            }
        }
    }

    private boolean injectFault(String targetClass) {

        // 在Soot.Scene中添加基础Class
        Scene.v().addBasicClass("java.io.FileWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.Writer", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.System", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.String", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.Long", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.OutputStreamWriter", SootClass.SIGNATURES);
//        add CustomActivationController class for CUSTOMIZED activationMode
        Scene.v().addBasicClass(runningParameters.getCustomizedActivation(), SootClass.SIGNATURES);

        logger.debug("Start to inject: " + runningParameters.getID() + " with " + runningParameters.getFaultType() + " into "
                + targetClass);
        BodyTransformer transformer = TransformerHelper.getTransformer(runningParameters.getFaultType(), runningParameters); // 根据运行时参数，构建故障注入用的BodyTransformer对象
        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", transformer)); // 利用PackManager，在jtp中添加故障注入用的BodyTransformer对象，使得之后的Soot.Main()将会执行该BodyTransformer对象

        try {
            MainWrapper.main(this.buildArgs(targetClass)); // 构建Soot.Main()的参数，并调用Soot.Main()开始注入
            if (runningParameters.isInjected()) { // 若注入成功
                logger.debug("Succeed to inject:" + runningParameters.getID() +
                        " with " + runningParameters.getFaultType() + " into " + targetClass);
                return true;
            } else { // 若注入失败
                logger.debug("Failed to inject:" + runningParameters.getID() +
                        " with " + runningParameters.getFaultType() + " into " + targetClass);
                return false;
            }

        } catch (Exception e) { // 若注入过程中发生异常
            logger.debug("Exception during injecting:" + runningParameters.getID() +
                    " with " + runningParameters.getFaultType() + " into " + targetClass, e);
            return false;
        } finally {
            PackManager.v().getPack("jtp").remove("jtp.instrumenter"); // 移除jtp中用于本次故障注入的BodyTransformer
            G.reset(); // 重置G
        }
    }

    // 构建Soot.Main()的参数
    private String[] buildArgs(String classWithPackage) {
        String[] args;
        if (this.debugMode) {
            args = new String[12];
            args[0] = "-cp";
            args[1] = ".:" + inputPath;
            args[2] = "-pp";
            args[3] = "-p";
            args[4] = "jb";
            args[5] = "use-original-names:true";
            args[6] = "-f";
            args[7] = "jimple";
            args[8] = "-d";
            // Store output files according to faultID
            args[9] = outputPath + File.separator + runningParameters.getID();
            args[10] = "-keep-line-number";
            args[11] = classWithPackage;
        } else {
            args = new String[10];
            args[0] = "-cp";
            String fullClassPath = getAllClassPath(inputPath);
            args[1] = fullClassPath;
            // logger.error(fullClassPath);
            args[2] = "-pp";
            args[3] = "-p";
            args[4] = "jb";
            args[5] = "use-original-names:true";
            args[6] = "-d";
            // Store output files according to faultID
            args[7] = outputPath + File.separator + runningParameters.getID();
            args[8] = "-keep-line-number";
            args[9] = classWithPackage;
        }
        return args;
    }

    // to get all the classpaths for the dependency of the class file to be injected
    private String getAllClassPath(String inputPath) {
        StringBuilder fullClassPath = new StringBuilder(".");
        File rootPack = new File(this.dependencyPath);
        if (rootPack.exists() && rootPack.isDirectory()) {
            File[] files = rootPack.listFiles();
            assert files != null;
            for (File tmpFile : files) {
                if (tmpFile.isDirectory()) {
                    fullClassPath.append(":").append(tmpFile.getAbsolutePath());
                }
            }
            return fullClassPath.toString();
        } else {
            return fullClassPath.append(":").append(inputPath).toString();
        }
    }

    // 获取全部的class名
    private List<String> getFullClassName() {
        String rootPackage;
        if (subPackage == null || subPackage.equals("")) {
            rootPackage = inputPath;
        } else {
//            Get class file directory by package name
            rootPackage = inputPath + File.separator + subPackage.replace(".", File.separator);
        }

        File rootPack = new File(rootPackage);
        if (rootPack.exists() && rootPack.isDirectory()) {
            return this.getQualifiedFullClass(rootPack);
        }
        return null;

    }

    // 获取全部符合要求的class名
    private List<String> getQualifiedFullClass(File folder) {
        List<String> allFullClassName = new ArrayList<String>();
        File[] files = folder.listFiles();
        assert files != null;
        for (File tmpFile : files) {
            if (tmpFile.isDirectory()) {
                allFullClassName.addAll(this.getQualifiedFullClass(tmpFile));
            } else {
                if (className == null) {
                    className = "";
                }
                if (tmpFile.getAbsolutePath().endsWith(className + ".class")) {
                    allFullClassName.add(tmpFile.getAbsolutePath());
                }
            }
        }
        return allFullClassName;
    }
}
