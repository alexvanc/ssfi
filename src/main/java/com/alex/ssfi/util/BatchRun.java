package com.alex.ssfi.util;

import java.util.List;

public class BatchRun {
    private int counter;
    private String distributionMode;
    private List<SingleRun> instances;
    public int getCounter() {
        return counter;
    }
    public void setCounter(int counter) {
        this.counter = counter;
    }
    public String getDistributionMode() {
        return distributionMode;
    }
    public void setDistributionMode(String distributionMode) {
        this.distributionMode = distributionMode;
    }
    public List<SingleRun> getInstances() {
        return instances;
    }
    public void setInstances(List<SingleRun> instances) {
        this.instances = instances;
    } 

}
