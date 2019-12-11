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
import soot.Local;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class ExceptionUncaughtTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(ExceptionUncaughtTransformer.class);

	public ExceptionUncaughtTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "EXCEPTION_UNCAUGHT_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// finally perform injection
		if (this.inject(b)) {
			this.parameters.setInjected(true);
			this.recordInjectionInfo();
			logger.debug("Succeed to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
					+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
			return;
		} else {
			logger.debug("Failed injection:+ " + this.formatInjectionInfo());
		}

		logger.debug("Fail to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private boolean inject(Body b) {
		// we directly throw a Exception instance at the entrance of the method
		// we don't distinguish try-catch, throw, none of the two anymore
		try {
			Chain<Unit> units = b.getUnits();
			List<Stmt> stmts = this.getUncaughtStatments(b);
			Unit firstUnit = units.getFirst();
			for (int i = 0; i < stmts.size(); i++) {
				if (i == 0) {
					// here we add the exception before the first statement of this method
					units.insertBefore(stmts.get(i), firstUnit);
				} else {
					units.insertAfter(stmts.get(i), stmts.get(i - 1));
				}
			}
			List<Stmt> preStmts = this.getPrecheckingStmts(b);
			for (int i = 0; i < preStmts.size(); i++) {
				if (i == 0) {
					units.insertBefore(preStmts.get(i), stmts.get(0));
				} else {
					units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
				}
			}
			List<Stmt> conditionStmts = this.getConditionStmt(b, firstUnit);
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
			return true;
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.formatInjectionInfo());
			return false;
		}
	}

	private List<Stmt> getUncaughtStatments(Body b) {
		// Directly throw a new Exception instance
		SootClass rootExceptionClass = Scene.v().getSootClass("java.lang.Exception");
		// Find the constructor of this Exception
		SootMethod constructor = null;
		Iterator<SootMethod> smIt = rootExceptionClass.getMethods().iterator();
		while (smIt.hasNext()) {
			SootMethod tmp = smIt.next();
			String signature = tmp.getSignature();
			// TO-DO
			// we should also decide which constructor should be used
			if (signature.contains("void <init>(java.lang.String)")) {
				constructor = tmp;
			}
		}
		// create an exception and initialize it
		Local lexception = Jimple.v().newLocal("tmpException", RefType.v(rootExceptionClass));
		b.getLocals().add(lexception);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception,
				Jimple.v().newNewExpr(RefType.v(rootExceptionClass)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef(),
				StringConstant.v("Fault Injection")));

		ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);
		List<Stmt> stmts = new ArrayList<Stmt>();
		stmts.add(assignStmt);
		stmts.add(invStmt);
		stmts.add(throwStmt);
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
			boolean noExceptionDeclared = true;
			boolean noExceptionCaught = true;
			List<SootClass> declaredExcepts = method.getExceptions();
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
			Iterator<Trap> caughtExcepts = tmpBody.getTraps().snapshotIterator();
			for (int j = 0, size = declaredExcepts.size(); j < size; j++) {
				SootClass exception = declaredExcepts.get(j);
				if (exception.getName().equals("java.lang.Exception")) {
					noExceptionDeclared = false;
					break;
				}
			}
			while (caughtExcepts.hasNext()) {
				Trap trap = caughtExcepts.next();
				// actually it's not that accurate
				// multiple try-catch blocks and multiple catch blocks for one try-catch is the
				// same in soot
				SootClass exception = trap.getException();
				if (exception.getName().equals("java.lang.Exception")) {
					noExceptionDeclared = false;
					break;
				}
			}
			if ((!noExceptionDeclared) || (!noExceptionCaught)) {
				// Exception are either declared or caught in this method
				continue;
			}

			if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>"))&& (!method.getName().contains("<clinit>"))) {
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
