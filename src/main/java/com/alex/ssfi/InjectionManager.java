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

    private InjectionManager(Configuration config) {
        this.debugMode = config.isDebug();
        this.inputPath = config.getInputPath();
        this.outputPath = config.getOutputPath();
        this.dependencyPath = config.getDependencyPath();
        this.subPackage = config.getPackagePattern();
        this.className = config.getClassPattern();
        this.runningParameters = new RunningParameter(config);
    }

    public static InjectionManager getManager(Configuration config) {
        if (instance == null) {
            instance = new InjectionManager(config);
        }
        return instance;
    }

    public boolean startInjection() {
        List<String> allClassName = this.getFullClassName();

        if (allClassName == null || allClassName.isEmpty()) {
            logger.error("Failed to find a possible targeting class file!");
            return false;
        }
        logger.debug("Prepare to inject one fault with possible classes number" + allClassName.size());
        while (true) {
            if (allClassName.isEmpty()) {
                // generate unqualified report
                logger.error("Failed to find a qualified class to inject a fault!");
                return false;
            }

            int index = new Random().nextInt(allClassName.size());
            String targetPath = allClassName.remove(index);

            int start = targetPath.indexOf(inputPath) + inputPath.length() + 1;
            int end = targetPath.indexOf(".class");
            String classWithPackage = targetPath.substring(start, end).replace(File.separatorChar, '.');


            if (this.injectFault(classWithPackage)) {
                logger.info("Succeed to inject one fault");
                return true;
            }
        }
    }

    private boolean injectFault(String targetClass) {

        Scene.v().addBasicClass("java.io.FileWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.Writer", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.System", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.String", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.Long", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.OutputStreamWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass(runningParameters.getCustomizedActivation(), SootClass.SIGNATURES);

        logger.debug("Start to inject: " + runningParameters.getID() + " with " + runningParameters.getFaultType() + " into "
                + targetClass);
        BodyTransformer transformer = TransformerHelper.getTransformer(runningParameters.getFaultType(), runningParameters);
        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", transformer));

        try {
            MainWrapper.main(this.buildArgs(targetClass));
            if (runningParameters.isInjected()) {
                logger.debug("Succeed to inject:" + runningParameters.getID() +
                        " with " + runningParameters.getFaultType() + " into " + targetClass);
                return true;
            } else {
                logger.debug("Failed to inject:" + runningParameters.getID() +
                        " with " + runningParameters.getFaultType() + " into " + targetClass);
                return false;
            }

        } catch (Exception e) {
            logger.debug("Exception during injecting:" + runningParameters.getID() +
                    " with " + runningParameters.getFaultType() + " into " + targetClass);
            logger.error(e.getMessage());
            return false;
        } finally {
            PackManager.v().getPack("jtp").remove("jtp.instrumenter");
            G.reset();
        }
    }

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
            // args[6] = "-w";
            // args[7] = "-f";
            // args[8] = "jimple";
            // args[9] = "-d";
            // args[10] = output;
            // args[11] = classWithPackage;
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
            // args[6] = "-w";

            // args[7] = "-d";
            // args[8] = output;
            // args[9] = classWithPackage;
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
            return fullClassPath + inputPath;
        }
    }

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

    /*
     * Summarize the fault injection experiments
     */
    private void generateReport() {
        // TODO
    }

}
