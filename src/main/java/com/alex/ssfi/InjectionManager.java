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
    private static final Logger logger = LogManager.getLogger(InjectionManager.class);
    // private static final Logger recorder=LogManager.getLogger("inject_recorder");

    private static InjectionManager instance = new InjectionManager();
    private boolean debugMode = false;
    private String dependencyPath;

    private InjectionManager() {
    }

    public static InjectionManager getManager() {
        return instance;
    }

    public boolean peformInjection(Configuration config) {
        this.debugMode = config.isDebug();
        this.dependencyPath = config.getDependencyPath();
        String inputPath = config.getInputPath();
        List<String> allClassName;
        allClassName = this.getFullClassName(inputPath, config.getPackagePattern(), config.getClassPattern());

        if ((allClassName == null) || allClassName.size() == 0) {
            logger.error("Failed to find a posssible targeting class file!");
            return false;
        }
        logger.debug("Prepare to inject one fault with possible classes number" + allClassName.size());
        while (true) {
            int length = allClassName.size();
            if (length == 0) {
                // generate unqualified report
                logger.error("Failed to find a qualified class to inject a fault!");
                return false;
            }
            int index = new Random().nextInt(length);

            String targetPath = allClassName.get(index);
            allClassName.remove(index);
            int start = targetPath.indexOf(inputPath) + inputPath.length() + 1;
            int end = targetPath.indexOf(".class");
            String classWithPackage = targetPath.substring(start, end).replace(File.separatorChar, '.');
            RunningParameter parameters = new RunningParameter(config, classWithPackage);
            if (this.injectFault(parameters)) {
                logger.info("Succeed to inject one fault");
                return true;
            }
        }
    }

    private boolean injectFault(RunningParameter parameters) {

        Scene.v().addBasicClass("java.io.FileWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.Writer", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.System", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.String", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.lang.Long", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.OutputStreamWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass(parameters.getCustomizedActivation(), SootClass.SIGNATURES);

        logger.debug("Start to inject: " + parameters.getID() + " with " + parameters.getFaultType() + " into "
                + parameters.getClassWithPackage());
        BodyTransformer transformer = TransformerHelper.getTransformer(parameters.getFaultType(), parameters);
        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", transformer));

        try {
            MainWrapper.main(this.buildArgs(parameters.getClassWithPackage(), parameters.getInput(),
                    // Store output files according to faultID
                    parameters.getOutput() + File.separator + parameters.getID()));
            if (parameters.isInjected()) {
                logger.debug("Succeed to inject:" + parameters.getID() + " with " + parameters.getFaultType() + " into "
                        + parameters.getClassWithPackage());
                return true;
            } else {
                logger.debug("Failed to inject:" + parameters.getID() + " with " + parameters.getFaultType() + " into "
                        + parameters.getClassWithPackage());
                return false;
            }

        } catch (Exception e) {
            logger.debug("Exception during injecting:" + parameters.getID() + " with " + parameters.getFaultType() + " into "
                    + parameters.getClassWithPackage());
            logger.error(e.getMessage());
            return false;
        } finally {
            PackManager.v().getPack("jtp").remove("jtp.instrumenter");
            G.reset();
        }
    }

    private String[] buildArgs(String classWithPackage, String input, String output) {
        String[] args;
        if (this.debugMode) {
            args = new String[12];

            args[0] = "-cp";
            args[1] = ".:" + input;
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
            args[9] = output;
            args[10] = "-keep-line-number";
            args[11] = classWithPackage;
        } else {

            args = new String[10];
            args[0] = "-cp";
            String fullClassPath = getAllClassPath(input);
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
            args[7] = output;
            args[8] = "-keep-line-number";
            args[9] = classWithPackage;
        }

        return args;
    }

    // to get all the classpaths for the dependency of the class file to be injected
    private String getAllClassPath(String inputPath) {
        String fullClassPath = ".";
        File rootPack = new File(this.dependencyPath);
        if ((!rootPack.exists()) || (!rootPack.isDirectory())) {
            return fullClassPath + inputPath;
        }
        File[] files = rootPack.listFiles();
        for (int i = 0; i < files.length; i++) {
            File tmpFile = files[i];
            if (tmpFile.isDirectory()) {
                fullClassPath += ":" + tmpFile.getAbsolutePath();
            }
        }
        return fullClassPath;
    }

    private List<String> getFullClassName(String input, String subPackage, String className) {
        // List<String> allFullClassName=new ArrayList<String>();
        String rootPackage = null;
        if ((subPackage == null) || (subPackage == "")) {
            rootPackage = input;
        } else {
            rootPackage = input + File.separator + subPackage.replace(".", File.separator);
        }

        // Check whether we can directly use apache with regex expression later
        // Collection files = FileUtils.listFiles(
        // dir,
        // new RegexFileFilter("^(.*?)"),
        // DirectoryFileFilter.DIRECTORY
        // );
        File rootPack = new File(rootPackage);
        if ((!rootPack.exists()) || (!rootPack.isDirectory())) {
            return null;
        }
        return this.getQualifiedFullClass(rootPack, className);

    }

    private List<String> getQualifiedFullClass(File folder, String className) {
        List<String> allFullClassName = new ArrayList<String>();
        File[] files = folder.listFiles();
        for (int i = 0; i < files.length; i++) {
            File tmpFile = files[i];
            // if
            // ((!tmpFile.isDirectory())&&(tmpFile.getAbsolutePath().endsWith(className+".class"))
            // {
            // allFullClassName.add(tmpF)
            // }
            if (!tmpFile.isDirectory()) {
                if (className == null) {
                    className = "";
                }
                if (tmpFile.getAbsolutePath().endsWith(className + ".class")) {
                    allFullClassName.add(tmpFile.getAbsolutePath());
                }
            } else {
                allFullClassName.addAll(this.getQualifiedFullClass(tmpFile, className));
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
