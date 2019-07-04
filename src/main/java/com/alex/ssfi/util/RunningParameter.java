package com.alex.ssfi.util;

public class RunningParameter {
//	public RuningParameter() {
//		
//	}
	private String ID;
	private String output;
	private String activationMode;
	private int activationRate;
	private String variableScope;
	private String variableType;
	private String variableName;
	private String variableValue;
	private String methodName;
	private String action;
	private boolean injected=false;
	
	public RunningParameter(SingleRun singleRun,String output,String activationMode,int activationRate) {
		this.ID=IDHelper.generateID(20);
		this.output=output;
		this.activationMode=activationMode;
		this.activationRate=activationRate;
		this.methodName=singleRun.getMethodPattern();
		this.variableScope=singleRun.getVariableScope();
		this.variableType=singleRun.getVariableType();
		this.variableName=singleRun.getVariablePattern();
		this.variableValue=singleRun.getTargetValue();
		this.action=singleRun.getAction();
//		int faultTypeID=FaultTypeHelper.getValueTypeIDByName(singleRun.getType());
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
	public String getMethodName() {
		return methodName;
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

}
