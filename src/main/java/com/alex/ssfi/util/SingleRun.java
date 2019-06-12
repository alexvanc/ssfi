package com.alex.ssfi.util;

public class SingleRun {
    private String type;
    private LocaltionPattern locationPattern;
    private String variableType;
    private String variableScope;
    private String action;
    private String targetValue;
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

	public String getVariableScope() {
		return variableScope;
	}

	public void setVariableScope(String variableScope) {
		this.variableScope = variableScope;
	}

	public String getTargetValue() {
		return targetValue;
	}

	public void setTargetValue(String targetValue) {
		this.targetValue = targetValue;
	}


}
class LocaltionPattern{
    private String packageP;
    private String classP;
    private String methodP;
    private String variableP;
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
	public String getVariableP() {
		return variableP;
	}
	public void setVariableP(String variableP) {
		this.variableP = variableP;
	} 
    
}
