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

public class ConditionInversedTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);
	
	public ConditionInversedTransformer(RunningParameter parameters) {
		this.parameters=parameters;
		
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
//		this.injectInfo.put("FaultType", "CONDITION_INVERSED_FAULT");
//		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
//		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
//
//		Chain<Unit> units=b.getUnits();
//		Iterator<Unit> unitItr=units.iterator();
//		// choose one scope to inject faults
//		// if not specified, then randomly choose
//		while (unitItr.hasNext()) {
//			Stmt stmt=(Stmt)unitItr.next();
//			if(stmt instanceof IfStmt) {
//				if(this.inject(b,stmt)) {
//					List<Stmt> actStmts = this.createActivateStatement(b);
//					for (int i = 0; i < actStmts.size(); i++) {
//						if (i == 0) {
//							units.insertBefore(actStmts.get(i), stmt);
//						} else {
//							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
//						}
//					}
//					this.parameters.setInjected(true);
//					this.recordInjectionInfo();
//					break;
//				}
//			}
//		}

	}
	
	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		this.foundTargetMethod = false;
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "CONDITION_INVERSED_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject CONDITION_INVERSED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

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
		Chain<Unit> units=b.getUnits();
		try {
		IfStmt ifStmt=(IfStmt)stmt;
		Expr expr=(Expr)ifStmt.getConditionBox().getValue();
		if(expr instanceof GtExpr ) {
			GtExpr gtExp=(GtExpr)expr;
			LtExpr ltExp=Jimple.v().newLtExpr(gtExp.getOp1(), gtExp.getOp2());
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
		}else if(expr instanceof GeExpr ) {
			GeExpr geExp=(GeExpr)expr;
			LeExpr leExp=Jimple.v().newLeExpr(geExp.getOp1(), geExp.getOp2());
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
		}else if(expr instanceof LtExpr ) {
			LtExpr ltExp=(LtExpr)expr;
			GtExpr gtExp=Jimple.v().newGtExpr(ltExp.getOp1(), ltExp.getOp2());
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
		}else if(expr instanceof LeExpr ) {
			List<Stmt> actStmts = this.createActivateStatement(b);
			for (int i = 0; i < actStmts.size(); i++) {
				if (i == 0) {
					units.insertBefore(actStmts.get(i), ifStmt);
				} else {
					units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
				}
			}
			LeExpr leExp=(LeExpr)expr;
			GeExpr geExp=Jimple.v().newGeExpr(leExp.getOp1(), leExp.getOp2());
			ifStmt.setCondition(geExp);
			return true;
		}
		}catch (Exception e) {
			logger.error(e.getMessage());
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
