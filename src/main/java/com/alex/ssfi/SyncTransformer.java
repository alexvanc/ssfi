package com.alex.ssfi;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.Modifier;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.EnterMonitorStmt;
import soot.jimple.ExitMonitorStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.Stmt;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

/*
 * TODO
 * distinguish between static field and instance field
 * report the Soot bugs of failing to get the fields of a class
 */
public class SyncTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(SyncTransformer.class);

	public SyncTransformer(RunningParameter parameters) {
		super(parameters);

	}

	public SyncTransformer() {
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
		this.injectInfo.put("FaultType", "Value_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		List<String> scopes = getSyncScope(this.parameters.getVariableScope());

		logger.debug("Try to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// randomly combine scope, type , variable, action
		while (true) {
			int scopesSize = scopes.size();
			if (scopesSize == 0) {
				logger.debug("Cannot find qualified scopes");
				break;
			}
			int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize);
			String scope = scopes.get(scopeIndex);
			scopes.remove(scopeIndex);
			this.injectInfo.put("VariableScope", scope);
			// injection sync fault for methods
			List<String> actions = getPossibleActionForScope(scope, this.parameters.getAction());
			while (true) {
				int actionSize = actions.size();
				if (actionSize == 0) {
					break;
				}
				int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);
				String action = actions.get(actionIndex);
				this.injectInfo.put("Action", action);

				if (this.inject(b, scope, action)) {
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
					logger.debug("Succeed to inject SYNC_FAULT into " + this.injectInfo.get("Package") + " "
							+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
					return;
				} else {
					logger.debug("Failed injection:+ " + this.formatInjectionInfo());
				}
			}
		}

		logger.debug("Fail to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private boolean inject(Body b, String scope, String action) {
		if (scope.equals("method")) {
			if (this.injectMethodWithAction(b, action)) {
				return true;
			}

		} else {// block
			if (this.injectBlockWithAction(b, action)) {
				return true;
			}
		}
		return false;
	}

	private boolean injectBlockWithAction(Body b, String action) {
		logger.info(this.formatInjectionInfo());
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitIt = units.snapshotIterator();
		boolean started = false;
		while (unitIt.hasNext()) {// delete the first synchronized block
			Stmt tmpStmt = (Stmt) unitIt.next();
			if ((started) && (tmpStmt instanceof IdentityStmt)) {
				Stmt nextStmt = (Stmt) units.getSuccOf((Unit) tmpStmt);
				Stmt nextNextStmp=(Stmt) units.getSuccOf(nextStmt);
				if ((nextStmt instanceof ExitMonitorStmt) && (nextNextStmp instanceof ThrowStmt)) {
					Iterator<Trap> trapsItr = b.getTraps().snapshotIterator();
					while(trapsItr.hasNext()) {
						Trap tmpTrap=trapsItr.next();
						Unit startUnit=tmpTrap.getHandlerUnit();
						if (startUnit.equals(tmpStmt)){
							b.getTraps().remove(tmpTrap);
						}
					}
					units.remove(tmpStmt);
					units.remove(nextStmt);
					units.remove(nextNextStmp);
					return true;
				}
			} else if ((started) && (tmpStmt instanceof ExitMonitorStmt)) {
				units.remove(tmpStmt);
				
			}
			if (tmpStmt instanceof EnterMonitorStmt) {
				units.remove(tmpStmt);
				started = true;
			}

		}
		return false;
	}

	private boolean injectMethodWithAction(Body b, String action) {
		logger.info(this.formatInjectionInfo());
		SootMethod targetMethod = b.getMethod();
		if (targetMethod.isSynchronized()) {
			if (action.equals("ASYNC")) {
				int originalModifiers = targetMethod.getModifiers();
				int targetModifiers = originalModifiers & (~Modifier.SYNCHRONIZED);
				targetMethod.setModifiers(targetModifiers);
				System.out.println(targetMethod.isStatic());
				return true;
			}
		} else {
			if (action.equals("SYNC")) {
				int originalModifiers = targetMethod.getModifiers();
				int targetModifiers = originalModifiers | (Modifier.SYNCHRONIZED);
				targetMethod.setModifiers(targetModifiers);
				System.out.println(targetMethod.isStatic());
				return true;
			}
		}
		return false;
	}

	private List<String> getPossibleActionForScope(String scope, String speficifiedAction) {
		// For simplicity, we don't check whether the action is applicable to this type
		// of variables
		// this should be done in the configuration validator
		List<String> possibleActions = new ArrayList<String>();
		String specifiedAction = this.parameters.getAction();
		if ((specifiedAction == null) || (specifiedAction == "")) {
			if (scope.equals("method")) {// another scope is block
				possibleActions.add("SYNC");
				possibleActions.add("ASYNC");
			} else {
				possibleActions.add("ASYNC");
			}
		} else {
			// here we believe users know how to specify right configurations
			possibleActions.add(specifiedAction);
		}
		return possibleActions;
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

	private List<String> getSyncScope(String variableScope) {
		List<String> scopes = new ArrayList<String>();
		if ((variableScope == null) || (variableScope == "")) {
			scopes.add("method");
			scopes.add("block");
		} else {
			scopes.add(variableScope);
		}
		return scopes;
	}
}
