package com.alex.ssfi;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.SootMethod;
import soot.Unit;
import soot.jimple.InvokeStmt;
import soot.jimple.Stmt;
import soot.util.Chain;

public class InvokeRemovedTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(InvokeRemovedTransformer.class);

	public InvokeRemovedTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub

		this.methodIndex++;
		if (this.parameters.isInjected()) {
			return;
		}
		String methodName = b.getMethod().getName();
		String methodSubSignature = b.getMethod().getSubSignature();
		String specifiedMethodName = this.parameters.getMethodName();
		if ((specifiedMethodName == null) || (specifiedMethodName == "")) {// in the random method mode
			if (!this.foundTargetMethod) {
				// randomly generate a target method
				this.generateTargetMethod(b);
			}
			if (methodSubSignature.equals(this.targetMethodSubSignature)) {
				this.startToInject(b);
			} else {
				return;
			}
		} else {// in the customized method mode
			if (methodName.equalsIgnoreCase(specifiedMethodName)) {
				this.startToInject(b);
			} else {
				return;
			}
		}


	}

	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		this.foundTargetMethod = false;
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "UNUSED_INVOKE_REMOVED_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// finally perform injection
		if (this.inject(b)) {
			this.parameters.setInjected(true);
			this.recordInjectionInfo();
			logger.debug("Succeed to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
					+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
			return;
		} else {
			logger.debug("Failed injection:+ " + this.formatInjectionInfo());
		}

		logger.debug("Fail to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private void generateTargetMethod(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		if (this.methodIndex >= allMethods.size()) {
			return;
		}
		int targetMethodIndex = new Random(System.currentTimeMillis())
				.nextInt(allMethods.size() - this.methodIndex + 1);
		this.foundTargetMethod = true;
		this.targetMethodSubSignature = allMethods.get(this.methodIndex + targetMethodIndex - 1).getSubSignature();
		return;
	}

	private boolean inject(Body b) {
		// actually there is only one type of action
		// the action parameter is for future use
		try {
			Chain<Unit> units = b.getUnits();
			Iterator<Unit> unitItr = units.iterator();
			while (unitItr.hasNext()) {
				Stmt stmt = (Stmt) unitItr.next();
				// Currently we directly choose the first invokeStmt to delete
				if (stmt instanceof InvokeStmt) {
					InvokeStmt invStmt = (InvokeStmt) stmt;
//					Unit nextStmt=unitItr.next();
					List<Stmt> actStmts = this.createActivateStatement(b);
					for (int i = 0; i < actStmts.size(); i++) {
						if (i == 0) {
							units.insertBefore(actStmts.get(i), invStmt);
						} else {
							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
						}
					}
					units.remove(invStmt);
					this.recordInjectionInfo();
					this.parameters.setInjected(true);
					return true;
				}
			}

			return false;

		} catch (Exception e) {
			logger.error(e.getMessage());
			return false;
		}
	}




}
