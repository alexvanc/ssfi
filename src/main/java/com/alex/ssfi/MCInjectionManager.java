package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.BatchRun;
import com.alex.ssfi.util.Configuration;
import com.alex.ssfi.util.JobHelper;
import com.alex.ssfi.util.RunningParameter;
import com.alex.ssfi.util.SingleRun;

import soot.BodyTransformer;
import soot.G;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.Transform;

//mainly for Hadoop injection experiments
public class MCInjectionManager {
    private static final Logger logger = LogManager.getLogger(InjectionManager.class);
    // private static final Logger
    // recorder=LogManager.getLogger("inject_recorder");

    private static MCInjectionManager instance = new MCInjectionManager();
    private boolean debugMode = false;
    private long jobTimeoutValue;
    private String targetComponent;
    private String targetJar;
    private String dependencyPath;

    private MCInjectionManager() {
    }

    public static MCInjectionManager getManager() {
        return instance;
    }

    public void peformInjection(Configuration config) {
        this.debugMode = config.isDebug();
        this.jobTimeoutValue = config.getTimeout();
        this.targetComponent = config.getComponent();
        this.dependencyPath = config.getDependencyPath();
        if (config.getInjectionMode().equals("batch")) {
            logger.info("In Batch mode");

            this.performBatchInjection(config.getBatchRun(), config.getInputPath(), config.getOutputPath(),
                    config.getActivationMode(), config.getActivationRate());
        } else {
            logger.info("In Single mode");
            this.performSingleInjection(config.getSingleRun(), config.getInputPath(), config.getOutputPath(),
                    config.getActivationMode(), config.getActivationRate());
        }
    }

    private void performSingleInjection(SingleRun singleInjection, String input, String ouput, String activationMode,
            int activationRate) {

        List<String> allComponents = this.getAllComponents(input, this.targetComponent);
        while (true) {
            int length = allComponents.size();
            if (length == 0) {
                // generate unqualified report
                logger.debug("Failed to inject one fault");
                return;
            }
            int index = new Random().nextInt(length);
            String componentName = allComponents.get(index);
            allComponents.remove(index);
            List<String> allJarFolders = this.getAllJarFiles(input, componentName, this.targetJar);
            while (true) {
                int jarLength = allJarFolders.size();
                if (jarLength == 0) {
                    break;
                }
                int jarIndex = new Random().nextInt(jarLength);
                String jarName = allJarFolders.get(jarIndex);
                allJarFolders.remove(jarIndex);
                String finalInput = input + File.separatorChar + componentName + File.separatorChar + jarName;
                List<String> allClassName;
                allClassName = this.getFullClassName(finalInput, singleInjection.getPackagePattern(),
                        singleInjection.getClassPattern());

                if ((allClassName == null) || allClassName.size() == 0) {
                    logger.error("Failed to find a posssible targeting class file!");
                    // generate unqualified report
                    continue;
                }
                logger.debug("Prepare to inject one fault with possible classes number" + allClassName.size());
                while (true) {
                    int classLength = allClassName.size();
                    if (classLength == 0) {
                        // generate unqualified report
                        logger.debug("Failed to inject one fault");
                        break;
                    }
                    int classIndex = new Random().nextInt(classLength);

                    String targetPath = allClassName.get(classIndex);
                    allClassName.remove(classIndex);
                    int start = targetPath.indexOf(finalInput) + finalInput.length() + 1;
                    int end = targetPath.indexOf(".class");
                    String classWithPackge = targetPath.substring(start, end).replace(File.separatorChar, '.');
                    // System.out.println(start+" "+classWithPackge);
                    if (this.injectFault(singleInjection, classWithPackge, input, componentName, jarName, ouput,
                            activationMode, activationRate)) {
                        logger.info("Succeed to inject one fault");
                        logger.info(classWithPackge);
                        return;
                    }
                }
            }

        }

    }

    private boolean injectFault(SingleRun singleInjection, String classWithPackage, String input, String componentName,
            String jarName, String output, String activationMode, int activationRate) {
        Scene.v().addBasicClass("java.io.FileWriter", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.Writer", SootClass.SIGNATURES);
        Scene.v().addBasicClass("java.io.OutputStreamWriter", SootClass.SIGNATURES);
        // Scene.v().loadNecessaryClasses();
        RunningParameter parameter = new RunningParameter(singleInjection, componentName, jarName, output,
                activationMode, activationRate);
        logger.debug("Start to inject: " + parameter.getID() + " with " + singleInjection.getType() + " into "
                + classWithPackage);
        BodyTransformer transformer = TransformerHelper.getTransformer(singleInjection.getType(), parameter);
        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", transformer));
        String finalInput = input + File.separatorChar + componentName + File.separatorChar + jarName;
        try {
            MainWrapper.main(this.buildArgs(classWithPackage, finalInput, output));
            if (parameter.isInjected()) {
                logger.debug("Succeed to inject:" + parameter.getID() + " with " + singleInjection.getType() + " into "
                        + classWithPackage);
                // Execute a job to observe the FI results
                JobHelper.runJob(parameter.getID(), classWithPackage, input, componentName, jarName, output,
                        this.jobTimeoutValue);
                return true;
            } else {
                logger.debug("Failed to inject:" + parameter.getID() + " with " + singleInjection.getType() + " into "
                        + classWithPackage);
                return false;
            }

        } catch (Exception e) {
            logger.error("Failed to inject2:" + parameter.getID() + " with " + singleInjection.getType() + " into "
                    + classWithPackage);
            logger.error(e.getMessage());
            return false;
        } finally {
            PackManager.v().getPack("jtp").remove("jtp.instrumenter");
            G.reset();
        }
    }

    private String[] buildArgs(String classWithPackage, String input, String output) {
        // TODO should guarantee input is the classPath
        String[] args;
        if (this.debugMode) {
            args = new String[11];

            args[0] = "-cp";
            args[1] = ".:" + input;
            args[2] = "-pp";
            args[3] = "-p";
            args[4] = "jb";
            args[5] = "use-original-names:true";
            args[6] = "-f";
            args[7] = "jimple";
            args[8] = "-d";
            args[9] = output;
            args[10] = classWithPackage;
        } else {
            args = new String[9];
            args[0] = "-cp";
            String fullClassPath = getAllClassPath(input);
            args[1] = fullClassPath;
            // logger.error(fullClassPath);
            args[2] = "-pp";
            args[3] = "-p";
            args[4] = "jb";
            args[5] = "use-original-names:true";

            args[6] = "-d";
            args[7] = output;
            args[8] = classWithPackage;
        }

        return args;
    }

    // this is only used for multi-components applications
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

    private List<String> getAllComponents(String input, String targetComponent) {
        // TODO Auto-generated method stub
        List<String> allComponents = new ArrayList<String>();
        File projectFolder = new File(input);
        if ((!projectFolder.exists()) || (!projectFolder.isDirectory())) {
            return allComponents;
        }
        File[] files = projectFolder.listFiles();
        for (int i = 0, size = files.length; i < size; i++) {
            if ((targetComponent == null) || (targetComponent == "")) {
                if (files[i].isDirectory()) {
                    allComponents.add(files[i].getName());
                }
            } else {
                if ((files[i].isDirectory()) && (files[i].getName().equals(targetComponent))) {
                    allComponents.add(files[i].getName());
                }
            }

        }
        return allComponents;
    }

    private List<String> getAllJarFiles(String input, String component, String targetJar) {
        List<String> allJars = new ArrayList<String>();
        File componentFolder = new File(input + File.separatorChar + component);
        if ((!componentFolder.exists()) || (!componentFolder.isDirectory())) {
            return allJars;
        }

        File[] files = componentFolder.listFiles();
        for (int i = 0, size = files.length; i < size; i++) {
            if ((targetJar == null) || (targetJar == "")) {
                if (files[i].isDirectory()) {
                    allJars.add(files[i].getName());
                }
            } else {
                if ((files[i].isDirectory()) && (files[i].getName().equals(targetJar))) {
                    allJars.add(files[i].getName());
                }
            }
        }
        return allJars;
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

    private void performBatchInjection(BatchRun batchInjection, String input, String ouput, String activationMode,
            int activationRate) {
        List<SingleRun> allRuns = batchInjection.getFaultList();

        String distributionMode = batchInjection.getDistributionMode();
        logger.info("Batch size: " + batchInjection.getCounter());
        if ((distributionMode == null) || distributionMode.equals("random")) {// random
                                                                              // distribution
                                                                              // mode
            int[] injectionNumber = this.randomDistribute(batchInjection.getCounter(), allRuns.size());
            logger.debug("Random distribution: " + injectionNumber.toString());
            for (int i = 0; i < allRuns.size(); i++) {
                for (int j = 0; j < injectionNumber[i]; j++) {
                    this.performSingleInjection(allRuns.get(i), input, ouput, activationMode, activationRate);
                }
            }
        } else if (distributionMode.equals("even")) {
            logger.debug("Even distribution: " + batchInjection.getCounter() / allRuns.size());
            for (int i = 0; i < allRuns.size(); i++) {
                int injectionNumber = batchInjection.getCounter() / allRuns.size();
                for (int j = 0; j < injectionNumber; j++) {
                    this.performSingleInjection(allRuns.get(i), input, ouput, activationMode, activationRate);
                }
            }
        } else {// in the fixed distribution mode
            for (int i = 0; i < allRuns.size(); i++) {
                SingleRun singleRun = allRuns.get(i);
                int injectionNumber = (int) (batchInjection.getCounter() * singleRun.getDistribution());
                for (int j = 0; j < injectionNumber; j++) {
                    this.performSingleInjection(singleRun, input, ouput, activationMode, activationRate);
                }
            }
        }
        generateReport();
    }

    private int[] randomDistribute(int total, int typeNumber) {
        int[] distribution = new int[typeNumber];
        Random rand = new Random();
        int seed = 100;
        for (int i = 0; i < (typeNumber - 1); i++) {
            int tmp = rand.nextInt(seed) + 1;
            distribution[i] = (int) (total * ((double) seed / (double) 100));
            seed -= tmp;
        }
        distribution[typeNumber - 1] = (int) (total * ((double) seed / (double) 100));
        return distribution;
    }

    /*
     * Summarize the fault injection experiments
     */
    private void generateReport() {
        // TODO
    }

    public String getTargetJar() {
        return targetJar;
    }

    public void setTargetJar(String targetJar) {
        this.targetJar = targetJar;
    }

    public String getDependencyPath() {
        return dependencyPath;
    }

    public void setDependencyPath(String dependencyPath) {
        this.dependencyPath = dependencyPath;
    }

}
