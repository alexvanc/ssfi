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
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.SwitchStmt;
import soot.util.Chain;

public class SwitchFallThroughTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(SwitchFallThroughTransformer.class);

	public SwitchFallThroughTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {

		while (!this.parameters.isInjected()) {
			// in this way, all the FIs are performed in the first function of this class
			SootMethod targetMethod = this.generateTargetMethod(b);
			if (targetMethod == null) {
				return;
			}
			this.startToInject(targetMethod.getActiveBody());
		}
	}

	private void startToInject(Body b) {

		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "SWITCH_FALLTHROUGH_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());
		List<String> actions = this.getTargetAction(this.parameters.getAction());

		logger.debug("Try to inject SWITCH_FALLTHROUGH_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// randomly combine action
		while (true) {
			int actionSize = actions.size();
			if (actionSize == 0) {
				logger.debug("Cannot find qualified actions");
				break;
			}
			int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);
			String action = actions.get(actionIndex);
			actions.remove(actionIndex);
			this.injectInfo.put("Action", action);
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
				if (this.inject(b, action, switchStmt)) {
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
					logger.debug("Succeed to inject SWITCH_FALLTHROUGH_FAULT into " + this.injectInfo.get("Package")
							+ " " + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
					return;
				} else {
					logger.debug("Failed injection:+ " + this.formatInjectionInfo());
				}
			}

			logger.debug("Fail to inject" + this.formatInjectionInfo());
		}

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

	private List<String> getTargetAction(String action) {
		List<String> actions = new ArrayList<String>();
		if ((action == null) || (action == "")) {
			actions.add("break");
			actions.add("fallthrough");
		} else {
			actions.add(action);
		}
		return actions;
	}

	private SootMethod generateTargetMethod(Body b) {
		if (this.allQualifiedMethods == null) {
			this.initAllQualifiedMethods(b);
		}
		int leftQualifiedMethodsSize = this.allQualifiedMethods.size();
		if (leftQualifiedMethodsSize == 0) {
			return null;
		}
		int randomMethodIndex = new Random(System.currentTimeMillis()).nextInt(leftQualifiedMethodsSize);
		SootMethod targetMethod = this.allQualifiedMethods.get(randomMethodIndex);
		this.allQualifiedMethods.remove(randomMethodIndex);
		return targetMethod;
	}

	private void initAllQualifiedMethods(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();
		boolean withSpefcifiedMethod = true;
		String specifiedMethodName = this.parameters.getMethodName();
		if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {
			withSpefcifiedMethod = false;
		}
		int length = allMethods.size();
		for (int i = 0; i < length; i++) {
			SootMethod method = allMethods.get(i);
			Iterator<Unit> units = method.getActiveBody().getUnits().snapshotIterator();
			while (units.hasNext()) {
				Unit unit = units.next();
				if (unit instanceof SwitchStmt) {
					if (!withSpefcifiedMethod) {
						allQualifiedMethods.add(method);
						break;
					} else {
						// it's strict, only when the method satisfies the condition and with the
						// specified name
						if (method.getName().equals(specifiedMethodName)) {// method names are strictly compared
							allQualifiedMethods.add(method);
							break;
						}
					}

				}
			}
		}
		this.allQualifiedMethods = allQualifiedMethods;
	}

	private boolean inject(Body b, String action, Stmt stmt) {
		if (action.equals("break")) {
			try {
				SwitchStmt switchStmt = (SwitchStmt) stmt;
				List<Unit> targets = switchStmt.getTargets();
				List<Integer> availableTargetIndex = new ArrayList<Integer>();

				// first we find the default target for the switch cases
				// the getDefaultTarget() method in soot is not what we want
				// meanwhile we find the cases can be use for injecting a break
				boolean foundBreakTarget = false;
				Unit breakStmt = null;
				Chain<Unit> units = b.getUnits();
				for (int j = 1; j < targets.size(); j++) {
					Unit caseBranch = targets.get(j);
					Unit stmtBeforeCase = units.getPredOf(caseBranch);
					if (stmtBeforeCase instanceof GotoStmt) {
						foundBreakTarget = true;
						breakStmt = stmtBeforeCase;
					} else {
						// find all the branches without a break stmt separating itself and next branch
						availableTargetIndex.add(j - 1);
					}
				}

				if (!foundBreakTarget) {// couldn't find destination for any break
					this.logger.debug(this.formatInjectionInfo());
					return false;
				}
				while (true) {
					int availableIndexSize = availableTargetIndex.size();
					if (availableIndexSize == 0) {
						return false;
					}
					int targetIndex = new Random(System.currentTimeMillis()).nextInt(availableIndexSize);
//					Unit currentTargetUnit = targets.get(availableTargetIndex.get(targetIndex));
					Unit nextTargetUnit = targets.get(availableTargetIndex.get(targetIndex) + 1);
					availableTargetIndex.remove(targetIndex);
					if (this.injectSwitchBreak(b, nextTargetUnit, breakStmt)) {
						return true;
					}
				}
			} catch (Exception e) {
				logger.error(e.getMessage());
				logger.error(this.injectInfo.toString());
				return false;
			}
		} else if (action.equals("fallthrough")) {

			try {
				SwitchStmt switchStmt = (SwitchStmt) stmt;
				List<Unit> targets = switchStmt.getTargets();
				List<Integer> availableTargetIndex = new ArrayList<Integer>();

				// first we find all the branches with a break stmt separating itself with next
				// branch
				Chain<Unit> units = b.getUnits();
				for (int j = 1; j < targets.size(); j++) {
					Unit caseBranch = targets.get(j);
					Unit stmtBeforeCase = units.getPredOf(caseBranch);
					if (stmtBeforeCase instanceof GotoStmt) {
						availableTargetIndex.add(j - 1);
					}
				}

				while (true) {
					int availableIndexSize = availableTargetIndex.size();
					if (availableIndexSize == 0) {
						return false;
					}
					int targetIndex = new Random(System.currentTimeMillis()).nextInt(availableIndexSize);
//					Unit currentTargetUnit = targets.get(availableTargetIndex.get(targetIndex));
					Unit nextTargetUnit = targets.get(availableTargetIndex.get(targetIndex) + 1);
					availableTargetIndex.remove(targetIndex);
					if (this.injectSwitchFallthrough(b, nextTargetUnit)) {
						return true;
					}
				}
			} catch (Exception e) {
				logger.error(e.getMessage());
				logger.error(this.injectInfo.toString());
				return false;
			}
		}
		return false;
	}

	// delete the break stmt which separates two different cases
	private boolean injectSwitchFallthrough(Body b, Unit nextTargetUnit) {
		// TODO Auto-generated method stub
		Chain<Unit> units = b.getUnits();
		Unit breakStmt = units.getPredOf(nextTargetUnit);

		List<Stmt> actStmts = this.createActivateStatement(b);
		for (int i = 0; i < actStmts.size(); i++) {
			if (i == 0) {
				units.insertBefore(actStmts.get(i), breakStmt);
			} else {
				units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
			}
		}
		units.remove(breakStmt);
		return true;

	}

	// add a break stmt between two cases without a break stmt
	private boolean injectSwitchBreak(Body b, Unit nextTargetUnit, Unit breakUnit) {
		Chain<Unit> units = b.getUnits();

		GotoStmt oriBreakStmt = (GotoStmt) breakUnit;
		GotoStmt breakStmt = Jimple.v().newGotoStmt(oriBreakStmt.getTarget());
		units.insertBefore(breakStmt, nextTargetUnit);
		List<Stmt> actStmts = this.createActivateStatement(b);
		for (int i = 0; i < actStmts.size(); i++) {
			if (i == 0) {
				units.insertBefore(actStmts.get(i), breakStmt);
			} else {
				units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
			}
		}
		return true;

	}

}
