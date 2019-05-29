package com.alex.ssfi.util;

public class SingleRun {
    private String faultType;
    private LocaltionPattern pattern;
    private String variableType;
    private String action;
    private String exceptionType;
    private String distribution;
        
    public String getFaultType() {
        return faultType;
    }
    public void setFaultType(String faultType) {
        this.faultType = faultType;
    }
    public String getVariableType() {
        return variableType;
    }
    public void setVariableType(String variableType) {
        this.variableType = variableType;
    }
    public String getAction() {
        return action;
    }
    public void setAction(String action) {
        this.action = action;
    }
    public LocaltionPattern getPattern() {
        return pattern;
    }
    public void setPattern(LocaltionPattern pattern) {
        this.pattern = pattern;
    }
    public String getExceptionType() {
        return exceptionType;
    }
    public void setExceptionType(String exceptionType) {
        this.exceptionType = exceptionType;
    }
    public String getDistribution() {
        return distribution;
    }
    public void setDistribution(String distribution) {
        this.distribution = distribution;
    }

}
class LocaltionPattern{
    private String packageP;
    private String classP;
    private String methodP;
    public String getPackageP() {
        return packageP;
    }
    public void setPackageP(String packageP) {
        this.packageP = packageP;
    }
    public String getClassP() {
        return classP;
    }
    public void setClassP(String classP) {
        this.classP = classP;
    }
    public String getMethodP() {
        return methodP;
    }
    public void setMethodP(String methodP) {
        this.methodP = methodP;
    } 
    
}
