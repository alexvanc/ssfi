package com.alex.ssfi.util;

public class Configuration {
    private String inputPath;
    private String outputPath;
    private String dependencyPath;
    private String activationLogFile;
    private String component;
    private String jarName;
    private boolean debug;
    private String activationMode = "ALWAYS";
    private int activationRate = 10;
    private int activationIndex = 1;
    private String type;
    private LocationPattern locationPattern;
    private String variableType;
    private String variableScope;
    private String action;
    private String targetValue;
    private String exceptionType;
    private float distribution;
    private String customizedActivation;

    // private static final Logger logger =
    // LogManager.getLogger(Configuration.class);

    public Configuration() {
    }

    public Configuration(String configFile) {

    }

    public String getCustomizedActivation() {
        return customizedActivation;
    }

    public void setCustomizedActivation(String customizedActivation) {
        this.customizedActivation = customizedActivation;
    }

    public int getActivationIndex() {
        return activationIndex;
    }

    public void setActivationIndex(int activationIndex) {
        this.activationIndex = activationIndex;
    }

    public String getType() {
        return type;

    }

    public void setType(String type) {
        this.type = type;
    }

    public LocationPattern getLocationPattern() {
        return locationPattern;
    }

    public void setLocationPattern(LocationPattern locationPattern) {
        this.locationPattern = locationPattern;
    }

    public String getVariableType() {
        return variableType;
    }

    public void setVariableType(String variableType) {
        this.variableType = variableType;
    }

    public String getVariableScope() {
        return variableScope;
    }

    public void setVariableScope(String variableScope) {
        this.variableScope = variableScope;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public String getTargetValue() {
        return targetValue;
    }

    public void setTargetValue(String targetValue) { this.targetValue = targetValue;}

    public String getExceptionType() {
        return exceptionType;
    }

    public void setExceptionType(String exceptionType) {
        this.exceptionType = exceptionType;
    }

    public float getDistribution() {
        return distribution;
    }

    public void setDistribution(float distribution) {
        this.distribution = distribution;
    }

    public boolean validateConfig() {
        // TODO
        // The normalization of the configuration file should be checked
        return true;
    }

    public String getInputPath() {
        return inputPath;
    }

    public void setInputPath(String inputPath) {
        this.inputPath = inputPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public void setOutputPath(String outputPath) {
        this.outputPath = outputPath;
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean isDebug() {
        return this.debug;
    }

    public String getActivationMode() {
        return activationMode;
    }

    public void setActivationMode(String activationMode) {
        this.activationMode = activationMode;
    }

    public int getActivationRate() {
        return activationRate;
    }

    public void setActivationRate(int activationRate) {
        this.activationRate = activationRate;
    }

    public String getComponent() {
        return component;
    }

    public void setComponent(String component) {
        this.component = component;
    }

    public String getJarName() {
        return jarName;
    }

    public void setJarName(String jarName) {
        this.jarName = jarName;
    }

    public String getDependencyPath() {
        return dependencyPath;
    }

    public void setDependencyPath(String dependencyPath) {
        this.dependencyPath = dependencyPath;
    }

    public String getActivationLogFile() {
        if (this.activationLogFile == null) {
            return "/tmp/injection.log";
        }
        return activationLogFile;
    }

    public void setActivationLogFile(String activationLogFile) {
        this.activationLogFile = activationLogFile;
    }

    public String getPackagePattern() {
        return this.locationPattern.getPackageP();
    }

    public String getClassPattern() {
        return this.locationPattern.getClassP();
    }

    public String getMethodPattern() {
        return this.locationPattern.getMethodP();
    }

    public String getVariablePattern() {
        return this.locationPattern.getVariableP();
    }

}

