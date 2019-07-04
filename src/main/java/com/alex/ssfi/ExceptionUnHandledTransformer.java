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
import soot.Trap;
import soot.Unit;
import soot.jimple.CaughtExceptionRef;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
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
		try {
			List<Trap> allTraps = this.getAllTraps(b);
			while (true) {
				int trapSize = allTraps.size();
				if (trapSize == 0) {
					return false;
				}
				int trapIndex = new Random(System.currentTimeMillis()).nextInt(trapSize);
				Trap targetTrap = allTraps.get(trapSize);
				allTraps.remove(trapIndex);
				if (this.injectUnhandled(b, targetTrap)) {
					return true;
				}
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.injectInfo.toString());
			return false;
		}
	}

	private boolean injectUnhandled(Body b, Trap trap) {
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitsItr = units.snapshotIterator();

		Unit beginUnit = trap.getHandlerUnit();
		GotoStmt gTryEndUnit = (GotoStmt) trap.getEndUnit();
		Unit outCatchUnit = gTryEndUnit.getTarget();
		// we delete all the process steps after this @caughtexception stmt
		while (unitsItr.hasNext()) {
			Stmt stmt = (Stmt) unitsItr.next();
			if (stmt.equals(beginUnit)) {// enter the target catch block
				while (unitsItr.hasNext()) {
					Stmt tmp = (Stmt) unitsItr.next();
					if (tmp.equals(outCatchUnit)) {
						// this try-catch only has one caught exception
						// or this is the last trap
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
						if (tmp instanceof IdentityStmt) {
							IdentityStmt tmpIDStmt = (IdentityStmt) tmp;
							if (tmpIDStmt.getRightOp() instanceof CaughtExceptionRef) {
								List<Stmt> actStmts = this.createActivateStatement(b);
								for (int i = 0; i < actStmts.size(); i++) {
									if (i == 0) {
										units.insertAfter(actStmts.get(i), beginUnit);
									} else {
										units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
									}
								}
								return true;
							}
						}
						units.remove(tmp);
					}
				}
			}

		}
		return false;

	}

	private List<Trap> getAllTraps(Body b) {
		List<Trap> allTraps = new ArrayList<Trap>();
		Iterator<Trap> trapItr = b.getTraps().snapshotIterator();
		while (trapItr.hasNext()) {
			allTraps.add(trapItr.next());
		}
		return allTraps;
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
			Chain<Trap> traps = method.getActiveBody().getTraps();
			if (traps.size() == 0) {
				continue;
			}

			if (!withSpefcifiedMethod) {
				allQualifiedMethods.add(method);
			} else {
				// it's strict, only when the method satisfies the condition and with the
				// specified name
				if (method.getName().equals(specifiedMethodName)) {// method names are strictly compared
					allQualifiedMethods.add(method);
				}
			}
		}

		this.allQualifiedMethods = allQualifiedMethods;
	}

}
