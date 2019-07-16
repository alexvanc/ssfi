package com.alex.ssfi.util;

import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Configuration {
	private String injectionMode;
	private String inputPath;
	private String outputPath;
	private boolean debug;
	private String activationMode = "always";
	private int activationRate = 10;
	private SingleRun singleRun;
	private BatchRun batchRun;

	private long timeout = 15000;

	private static final Logger logger = LogManager.getLogger(Configuration.class);

	public Configuration() {
	};

	public Configuration(String configFile) {

	}

	public boolean validateConfig() {
		if (this.injectionMode == null) {
			return false;
		}
		if (this.injectionMode.equals("batch")) {
			List<SingleRun> runs = batchRun.getFaultList();
			for (int i = 0; i < runs.size(); i++) {
				if (!runs.get(i).validate()) {
					return false;
				}
			}
			return true;
		} else if (this.injectionMode.equals("single")) {
			return this.singleRun.validate();
		} else {
			logger.error("Invalid injectionMode!");
			return false;
		}
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

	public long getTimeout() {
		return timeout;
	}

	public void setTimeout(long timeout) {
		this.timeout = timeout;
	}

}
