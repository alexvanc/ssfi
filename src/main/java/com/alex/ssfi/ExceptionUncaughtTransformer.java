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
//		this.injectInfo.put("FaultType", "EXCEPTION_UNCAUGHT_FAULT");
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
			for (int i = 0; i < stmts.size(); i++) {
				if (i == 0) {
					// here we add the exception before the first statement of this method
					units.insertBefore(stmts.get(i), units.getFirst());
				} else {
					units.insertAfter(stmts.get(i), stmts.get(i - 1));
				}
			}

			List<Stmt> actStmts = this.createActivateStatement(b);
			for (int i = 0; i < actStmts.size(); i++) {
				if (i == 0) {
					units.insertBefore(actStmts.get(i), stmts.get(0));
				} else {
					units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
				}
			}
			return true;
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.injectInfo.toString());
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
