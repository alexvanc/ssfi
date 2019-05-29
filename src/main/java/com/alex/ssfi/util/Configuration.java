package com.alex.ssfi.util;

public class Configuration {
    private String injectionMode;
    private String inputPath;
    private String outputPath;
    private SingleRun singleRun;
    private BatchRun batchRun;
    
    public Configuration(){};
    public Configuration(String configFile){
        
    }
    public String getInjectionMode() {
        return injectionMode;
    }
    public void setInjectionMode(String injectionMode) {
        this.injectionMode = injectionMode;
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
    public SingleRun getSingleRun() {
        return singleRun;
    }
    public void setSingleRun(SingleRun singleRun) {
        this.singleRun = singleRun;
    }
    public BatchRun getBatchRun() {
        return batchRun;
    }
    public void setBatchRun(BatchRun batchRun) {
        this.batchRun = batchRun;
    }

}
