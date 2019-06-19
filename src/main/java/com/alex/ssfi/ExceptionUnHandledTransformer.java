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
import soot.Trap;
import soot.Unit;
import soot.jimple.Stmt;
import soot.util.Chain;

//this type of fault only applies to try-catch block
public class ExceptionUnHandledTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(ValueTransformer.class);

	public ExceptionUnHandledTransformer(RunningParameter parameters) {
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

//		if (this.parameters.isInjected()) {
//			return;
//		}
//		String methodSignature = b.getMethod().getName();
//		String targetMethodName = this.parameters.getMethodName();
//		if ((targetMethodName != null) && (!targetMethodName.equalsIgnoreCase(methodSignature))) {
//			return;
//		}
//		this.injectInfo.put("FaultType", "EXCEPTION_UNHANDLED_FAULT");
//		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
//		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
//		List<String> scopes = getTargetScope(this.parameters.getVariableScope());
//
//		// choose one scope to inject faults
//		// if not specified, then randomly choose
//		while (true) {
//			int scopesSize = scopes.size();
//			if (scopesSize == 0) {
//				logger.debug("Cannot find qualified scopes");
//				break;
//			}
//			int scopeIndex = new Random().nextInt(scopesSize);
//			String scope = scopes.get(scopeIndex);
//			scopes.remove(scopeIndex);
//
//			if (this.inject(b, scope)) {
//				break;
//			}
//		}

	}

	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		this.foundTargetMethod = false;
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "EXCEPTION_UNHANDLED_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// finally perform injection
		if (this.inject(b)) {
			this.parameters.setInjected(true);
			this.recordInjectionInfo();
			logger.debug("Succeed to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
					+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
			return;
		} else {
			logger.debug("Failed injection:+ " + this.formatInjectionInfo());
		}

		logger.debug("Fail to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private boolean inject(Body b) {
		Chain<Unit> units = b.getUnits();
		Chain<Trap> traps = b.getTraps();
		if (traps.size() == 0) {
			return false;
		}
		int targetTrapIndex = new Random(System.currentTimeMillis()).nextInt(traps.size());
		Iterator<Trap> trapsItr = b.getTraps().snapshotIterator();
		Iterator<Unit> unitsItr=units.snapshotIterator();
		int index = 0;
		try {
			while (trapsItr.hasNext()) {
				Trap tmpTrap = trapsItr.next();
				if (index == targetTrapIndex) {
					Unit beginUnit = tmpTrap.getHandlerUnit();
					//we delete all the process steps after this @caughtexception stmt
					while (unitsItr.hasNext()) {
						Stmt stmt = (Stmt) unitsItr.next();
						if (stmt.equals(beginUnit)) {// enter the target catch block
							while (unitsItr.hasNext()) {
								Stmt tmp = (Stmt) unitsItr.next();
								//TODO check whether it satisfies all the conditions
								if (stmt.getBoxesPointingToThis().size()!=0) {// indicate this stmt is the end of catch block
									List<Stmt> actStmts = this.createActivateStatement(b);
									for (int i = 0; i < actStmts.size(); i++) {
										if (i == 0) {
											units.insertAfter(actStmts.get(i), beginUnit);
										} else {
											units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
										}
									}
									return true;
								} else {
									units.remove(tmp);// delete the units related with processing this caught exception
								}
							}
						}

					}

				} else {
					index++;
				}
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.injectInfo.toString());
			return false;
		}
		return false;
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

}
