package com.alex.ssfi.util;

public class SingleRun {
    private String type;
    private LocaltionPattern locationPattern;
    private String variableType;
    private String action;
    private String exceptionType;
    private float distribution;
    
    //TO DO
    //check conformation of configuraion rules
    public boolean validate() {
    	return true;
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
	public String getType() {
		return type;
	}
	public void setType(String type) {
		this.type = type;
	}
	public LocaltionPattern getLocationPattern() {
		return locationPattern;
	}
	public void setLocationPattern(LocaltionPattern locationPattern) {
		this.locationPattern = locationPattern;
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
