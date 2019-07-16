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
import soot.jimple.EqExpr;
import soot.jimple.Expr;
import soot.jimple.GeExpr;
import soot.jimple.GotoStmt;
import soot.jimple.GtExpr;
import soot.jimple.IfStmt;
import soot.jimple.Jimple;
import soot.jimple.LeExpr;
import soot.jimple.LtExpr;
import soot.jimple.NeExpr;
import soot.jimple.Stmt;
import soot.util.Chain;

public class ConditionInversedTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);

	public ConditionInversedTransformer(RunningParameter parameters) {
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
		// no matter this inject fails or succeeds, this targetMethod is already used
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "CONDITION_INVERSED_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject CONDITION_INVERSED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// actually we should do the randomization in the inject method
		// to keep coding stype the same
		List<Stmt> allIfStmt = getAllIfStmt(b);
		while (true) {
			int compStmtSize = allIfStmt.size();
			if (compStmtSize == 0) {
				break;
			}
			int randomCompIndex = new Random(System.currentTimeMillis()).nextInt(compStmtSize);
			Stmt targetCompStmt = allIfStmt.get(randomCompIndex);
			allIfStmt.remove(randomCompIndex);
			if (this.inject(b, targetCompStmt)) {
				this.parameters.setInjected(true);
				this.recordInjectionInfo();
				logger.debug("Succeed to inject CONDITION_INVERSED_FAULT into " + this.injectInfo.get("Package") + " "
						+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
				return;
			} else {
				logger.debug("Failed injection:+ " + this.formatInjectionInfo());
			}
		}

		// finally perform injection

		logger.debug("Fail to inject CONDITION_INVERSED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private boolean inject(Body b, Stmt stmt) {
		// TODO Auto-generated method stub
		Chain<Unit> units = b.getUnits();
		try {
			IfStmt ifStmt = (IfStmt) stmt;
			IfStmt newIfStmt = (IfStmt) ifStmt.clone();
			Unit nextUnit = units.getSuccOf(ifStmt);
			Expr expr = (Expr) newIfStmt.getConditionBox().getValue();
			if (expr instanceof GtExpr) {
				GtExpr gtExp = (GtExpr) expr;
				LeExpr leExp = Jimple.v().newLeExpr(gtExp.getOp1(), gtExp.getOp2());
				newIfStmt.setCondition(leExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);
				return true;
			} else if (expr instanceof GeExpr) {
				GeExpr geExp = (GeExpr) expr;
				LtExpr ltExp = Jimple.v().newLtExpr(geExp.getOp1(), geExp.getOp2());
				newIfStmt.setCondition(ltExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);
				return true;
			} else if (expr instanceof LtExpr) {
				LtExpr ltExp = (LtExpr) expr;
				GeExpr geExp = Jimple.v().newGeExpr(ltExp.getOp1(), ltExp.getOp2());
				newIfStmt.setCondition(geExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);
				return true;
			} else if (expr instanceof LeExpr) {
				LeExpr leExp = (LeExpr) expr;
				GtExpr gtExp = Jimple.v().newGtExpr(leExp.getOp1(), leExp.getOp2());
				newIfStmt.setCondition(gtExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);

				return true;
			} else if (expr instanceof EqExpr) {
				EqExpr eqExp = (EqExpr) expr;
				NeExpr neExp = Jimple.v().newNeExpr(eqExp.getOp1(), eqExp.getOp2());
				newIfStmt.setCondition(neExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);

				return true;
			} else if (expr instanceof NeExpr) {
				NeExpr neExp = (NeExpr) expr;
				EqExpr eqExp = Jimple.v().newEqExpr(neExp.getOp1(), neExp.getOp2());
				newIfStmt.setCondition(eqExp);
				units.insertBefore(newIfStmt, ifStmt);

				List<Stmt> preStmts = this.getPrecheckingStmts(b);
				for (int i = 0; i < preStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(preStmts.get(i), newIfStmt);
					} else {
						units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
					}
				}
				List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);
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
				GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);
				units.insertAfter(skipIfStmt, newIfStmt);

				return true;
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.formatInjectionInfo());
		}
		return false;
	}

	private List<Stmt> getAllIfStmt(Body b) {
		// TODO Auto-generated method stub
		List<Stmt> allCompareStmt = new ArrayList<Stmt>();
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitItr = units.iterator();
		// choose one scope to inject faults
		// if not specified, then randomly choose
		while (unitItr.hasNext()) {
			Stmt stmt = (Stmt) unitItr.next();
			if (stmt instanceof IfStmt) {
				allCompareStmt.add(stmt);

			}
		}
		return allCompareStmt;

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

	// for this fault type,we simply assume all methods satisfy the condition
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
			Iterator<Unit> unitItr = tmpBody.getUnits().snapshotIterator();
			while (unitItr.hasNext()) {
				Unit tmpUnit = unitItr.next();
				if (tmpUnit instanceof IfStmt) {
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

}
