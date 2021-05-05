package com.alex.ssfi.util;

public class RunningParameter {
	// public RuningParameter() {
	//
	// }
	private String ID;
	private String input;
	private String output;
	private String faultType;
	private String classWithPackage;
	private String activationMode;
	private String activationLogFile;
	private String componentName;
	private String jarName;
	private int activationRate;
	private int activationIndex;
	private String variableScope;
	private String variableType;
	private String variableName;
	private String variableValue;
	private String methodName;
	private String action;
	private boolean injected = false;

	public RunningParameter(SingleRun singleRun, String componentName, String jarName, String output,
			String activationMode, int activationRate) {
		this.ID = IDHelper.generateID(20);
		this.output = output;
		this.activationMode = activationMode;
		this.activationRate = activationRate;
		this.componentName = componentName;
		this.jarName = jarName;
		this.methodName = singleRun.getMethodPattern();
		this.variableScope = singleRun.getVariableScope();
		this.variableType = singleRun.getVariableType();
		this.variableName = singleRun.getVariablePattern();
		this.variableValue = singleRun.getTargetValue();
		this.action = singleRun.getAction();
		// int faultTypeID=FaultTypeHelper.getValueTypeIDByName(singleRun.getType());
	}

	public RunningParameter(Configuration config, String classWithPackage) {
		this.ID = IDHelper.generateID(20);
		this.faultType = config.getType();
		this.input = config.getInputPath();
		this.output = config.getOutputPath();
		this.activationMode = config.getActivationMode();
		this.activationRate = config.getActivationRate();
		this.activationIndex = config.getActivationIndex();
		this.methodName = config.getMethodPattern();
		this.variableScope = config.getVariableScope();
		this.variableType = config.getVariableType();
		this.variableName = config.getVariablePattern();
		this.variableValue = config.getTargetValue();
		this.action = config.getAction();
		this.activationLogFile = config.getActivationLogFile();
		this.classWithPackage = classWithPackage;
	}

	public String getID() {
		return ID;
	}

	public String getOutput() {
		return output;
	}

	public String getVariableType() {
		return variableType;
	}

	public String getVariableScope() {
		return variableScope;
	}

	public String getAction() {
		return action;
	}

	public String getVariableName() {
		return this.variableName;
	}

	public String toString() {
		return null;
	}

	public boolean isInjected() {
		return injected;
	}

	public void setInjected(boolean injected) {
		this.injected = injected;
	}

	public String getVariableValue() {
		return variableValue;
	}

	public String getActivationMode() {
		return activationMode;
	}

	public int getActivationRate() {
		return activationRate;
	}

	public String getComponentName() {
		return componentName;
	}

	public String getJarName() {
		return jarName;
	}

	public int getActivationIndex() {
		return activationIndex;
	}

	public String getActivationLogFile() {
		if (this.activationLogFile == null) {
			return "/tmp/activation.log";
		}
		return activationLogFile;
	}

	public void setActivationLogFile(String activationLogFile) {
		this.activationLogFile = activationLogFile;
	}

	public String getFaultType() {
		return faultType;
	}

	public String getClassWithPackage() {
		return classWithPackage;
	}

	public String getInput() {
		return input;
	}

	public String getMethodName() {
		return methodName;
	}	

}
