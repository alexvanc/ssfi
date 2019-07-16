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

public class SwitchMissDefaultTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(SwitchMissDefaultTransformer.class);

	public SwitchMissDefaultTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
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

	private synchronized SootMethod generateTargetMethod(Body b) {
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
			Body tmpBody;
			try {
				tmpBody = method.retrieveActiveBody();
			} catch (Exception e) {
				// currently we don't know how to deal with this case
				logger.info("Retrieve Body failed!");
				continue;
			}
			if (tmpBody == null) {
				continue;
			}
			Iterator<Unit> units = tmpBody.getUnits().snapshotIterator();
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

	private boolean inject(Body b, Stmt stmt) {
		// TODO Auto-generated method stub
		try {
			SwitchStmt switchStmt = (SwitchStmt) stmt;
			List<Unit> targets = switchStmt.getTargets();
			Unit defaultUnit = switchStmt.getDefaultTarget();
			// find a break target to directly go out switch block
			boolean foundBreakTarget = false;
			GotoStmt breakStmt = null;
			Chain<Unit> units = b.getUnits();
			for (int j = 1; j < targets.size(); j++) {
				Unit caseBranch = targets.get(j);
				Unit stmtBeforeCase = units.getPredOf(caseBranch);
				if (stmtBeforeCase instanceof GotoStmt) {
					foundBreakTarget = true;
					breakStmt = (GotoStmt) stmtBeforeCase;
					break;
				}
			}
			if (!foundBreakTarget) {// couldn't find destination for break
				this.logger.debug(this.formatInjectionInfo());
				return false;
			}
			if (breakStmt.getTarget().equals(defaultUnit)) {// there is no default block in this switch
				this.logger.debug(this.formatInjectionInfo());
				return false;
			}
			if (this.injectMissDefault(b, switchStmt, breakStmt, defaultUnit)) {
				return true;
			}

		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.formatInjectionInfo());
		}
		return false;

	}

	private boolean injectMissDefault(Body b, SwitchStmt switchStmt, GotoStmt breakDestinationStmt,
			Unit defaultTargetUnit) {
		Chain<Unit> units = b.getUnits();

		List<Stmt> preStmts = this.getPrecheckingStmts(b);
		for (int i = 0; i < preStmts.size(); i++) {
			if (i == 0) {
				units.insertBefore(preStmts.get(i), defaultTargetUnit);
			} else {
				units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
			}
		}
		List<Stmt> conditionStmts = this.getConditionStmt(b, defaultTargetUnit);
		for (int i = 0; i < conditionStmts.size(); i++) {
			if (i == 0) {
				units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
			} else {
				units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
			}
		}
		List<Stmt> actStmts = this.createActivateStatement(b);
		for (int i = 0; i < actStmts.size(); i++) {
			if (i == 0) {
				units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
			} else {
				units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
			}
		}

		GotoStmt outCatchStmt = Jimple.v().newGotoStmt(breakDestinationStmt.getTarget());
		units.insertAfter(outCatchStmt, actStmts.get(actStmts.size() - 1));
		switchStmt.setDefaultTarget(preStmts.get(0));

		return true;
	}

}
