package com.alex.ssfi.util;

import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Configuration {
	private String inputPath;
	private String outputPath;
	private String dependencyPath;
	private String activationLogFile;
	private String component;
	private String jarName;
	private boolean debug;
	private String activationMode = "always";
	private int activationRate = 10;
	private SingleRun singleRun;

	private static final Logger logger = LogManager.getLogger(Configuration.class);

	public Configuration() {
	};

	public Configuration(String configFile) {

	}

	public boolean validateConfig() {
		return this.singleRun.validate();
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


	public boolean isDebug() {
		return debug;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
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
		if (this.activationLogFile==null) {
			return "/tmp/injection.log";
		}
		return activationLogFile;
	}

	public void setActivationLogFile(String activationLogFile) {
		this.activationLogFile = activationLogFile;
	}

}

