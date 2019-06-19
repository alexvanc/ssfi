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
import soot.SootMethod;
import soot.Unit;
import soot.jimple.GotoStmt;
import soot.jimple.Stmt;
import soot.jimple.SwitchStmt;
import soot.util.Chain;

public class SwitchMissDefaultTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(SwitchMissDefaultTransformer.class);

	public SwitchMissDefaultTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "SWITCH_MISS_DEFAULT_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject SWITCH_MISS_DEFAULT_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		List<Stmt> allSwtichStmts = this.getAllSwitchStmt(b);
		while (true) {
			int stmtsSize = allSwtichStmts.size();
			if (stmtsSize == 0) {
				break;
			}
			int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtsSize);
			Stmt switchStmt = allSwtichStmts.get(stmtIndex);
			allSwtichStmts.remove(stmtIndex);

			// finally perform injection
			if (this.inject(b, switchStmt)) {
				this.parameters.setInjected(true);
				this.recordInjectionInfo();
				logger.debug("Succeed to inject SWITCH_MISS_DEFAULT_FAULT into " + this.injectInfo.get("Package") + " "
						+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
				return;
			} else {
				logger.debug("Failed injection:+ " + this.formatInjectionInfo());
			}
		}

		logger.debug("Fail to inject" + this.formatInjectionInfo());

	}

	private List<Stmt> getAllSwitchStmt(Body b) {
		// TODO Auto-generated method stub
		List<Stmt> stmts = new ArrayList<Stmt>();
		Iterator<Unit> unitItr = b.getUnits().snapshotIterator();
		while (unitItr.hasNext()) {
			Stmt tmpStmt = (Stmt) unitItr.next();
			if (tmpStmt instanceof SwitchStmt) {
				stmts.add(tmpStmt);
			}
		}
		return stmts;
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

	private boolean inject(Body b, Stmt stmt) {
		// TODO Auto-generated method stub
		try {
			SwitchStmt switchStmt = (SwitchStmt) stmt;
			List<Unit> targets = switchStmt.getTargets();
			List<Integer> availableTargetIndex = new ArrayList<Integer>();
			for (int i = 0; i < targets.size() - 1; i++) {
				availableTargetIndex.add(i);
			}
			// find a break target to directly go out switch
			boolean foundBreakTarget = false;
			Unit breakStmt = null;
			Chain<Unit> units = b.getUnits();
			for (int j = 1; j < targets.size(); j++) {
				Unit caseBranch = targets.get(j);
				Unit stmtBeforeCase = units.getPredOf(caseBranch);
				if (stmtBeforeCase instanceof GotoStmt) {
					foundBreakTarget = true;
					breakStmt = stmtBeforeCase;
				}
			}
			if (!foundBreakTarget) {// couldn't find destination for break
				this.logger.debug(this.formatInjectionInfo());
				return false;
			}
			
			GotoStmt breakDestination = (GotoStmt) breakStmt;
			Unit destinationUnit=breakDestination.getTarget();
		
			switchStmt.setDefaultTarget(breakDestination.getTarget());
			List<Stmt> actStmts = this.createActivateStatement(b);
			for (int i = 0; i < actStmts.size(); i++) {
				if (i == 0) {
					units.insertAfter(actStmts.get(i), destinationUnit);
				} else {
					units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
				}
			}
			return true;
		} catch (Exception e) {
			logger.error(e.getMessage());
		}
		return false;

	}

}
