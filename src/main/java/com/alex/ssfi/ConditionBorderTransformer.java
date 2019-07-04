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
import soot.jimple.Expr;
import soot.jimple.GeExpr;
import soot.jimple.GtExpr;
import soot.jimple.IfStmt;
import soot.jimple.Jimple;
import soot.jimple.LeExpr;
import soot.jimple.LtExpr;
import soot.jimple.Stmt;
import soot.util.Chain;

public class ConditionBorderTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(ConditionBorderTransformer.class);

	public ConditionBorderTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "CONDITION_BORDER_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		//actually we should do the randomization in the inject method 
				//to keep coding stype the same
		List<Stmt> allCompareStmt = getAllCompareStmt(b);
		while (true) {
			int compStmtSize = allCompareStmt.size();
			if (compStmtSize == 0) {
				break;
			}
			int randomCompIndex = new Random(System.currentTimeMillis()).nextInt(compStmtSize);
			Stmt targetCompStmt = allCompareStmt.get(randomCompIndex);
			allCompareStmt.remove(randomCompIndex);
			if (this.inject(b, targetCompStmt)) {
				this.parameters.setInjected(true);
				this.recordInjectionInfo();
				logger.debug("Succeed to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
						+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
				return;
			} else {
				logger.debug("Failed injection:+ " + this.formatInjectionInfo());
			}
		}

		// finally perform injection

		logger.debug("Fail to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private List<Stmt> getAllCompareStmt(Body b) {
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
			Iterator<Unit> unitItr = method.getActiveBody().getUnits().snapshotIterator();
			while (unitItr.hasNext()) {
				Unit tmpUnit = unitItr.next();
				if (tmpUnit instanceof IfStmt) {
					// we only deal with <, >, <=, >=
					IfStmt tmpIfStmt = (IfStmt) tmpUnit;
					Expr expr = (Expr) tmpIfStmt.getCondition();
					if ((expr instanceof GtExpr) || (expr instanceof GeExpr) || (expr instanceof LtExpr)
							|| (expr instanceof LeExpr)) {
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

		}

		this.allQualifiedMethods = allQualifiedMethods;
	}

	private boolean inject(Body b, Stmt stmt) {
		// TODO Auto-generated method stub
		Chain<Unit> units = b.getUnits();
		try {
			IfStmt ifStmt = (IfStmt) stmt;
			Expr expr = (Expr) ifStmt.getConditionBox().getValue();
			if (expr instanceof GtExpr) {
				GtExpr gtExp = (GtExpr) expr;
				GeExpr geExp = Jimple.v().newGeExpr(gtExp.getOp1(), gtExp.getOp2());
				ifStmt.setCondition(geExp);
				List<Stmt> actStmts = this.createActivateStatement(b);
				for (int i = 0; i < actStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(actStmts.get(i), ifStmt);
					} else {
						units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
					}
				}
				return true;
			} else if (expr instanceof GeExpr) {
				GeExpr geExp = (GeExpr) expr;
				GtExpr gtExp = Jimple.v().newGtExpr(geExp.getOp1(), geExp.getOp2());
				ifStmt.setCondition(gtExp);
				List<Stmt> actStmts = this.createActivateStatement(b);
				for (int i = 0; i < actStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(actStmts.get(i), ifStmt);
					} else {
						units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
					}
				}
				return true;
			} else if (expr instanceof LtExpr) {
				LtExpr ltExp = (LtExpr) expr;
				LeExpr leExp = Jimple.v().newLeExpr(ltExp.getOp1(), ltExp.getOp2());
				ifStmt.setCondition(leExp);
				List<Stmt> actStmts = this.createActivateStatement(b);
				for (int i = 0; i < actStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(actStmts.get(i), ifStmt);
					} else {
						units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
					}
				}
				return true;
			} else if (expr instanceof LeExpr) {
				LeExpr leExp = (LeExpr) expr;
				LtExpr ltExp = Jimple.v().newLtExpr(leExp.getOp1(), leExp.getOp2());
				ifStmt.setCondition(ltExp);
				List<Stmt> actStmts = this.createActivateStatement(b);
				for (int i = 0; i < actStmts.size(); i++) {
					if (i == 0) {
						units.insertBefore(actStmts.get(i), ifStmt);
					} else {
						units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
					}
				}
				return true;
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
		}
		return false;
	}

}
