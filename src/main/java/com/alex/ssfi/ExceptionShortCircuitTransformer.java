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

/*
 * TODO
 * decide parameter to choose which exception to throw or catch when there are multiple exceptions declared
 */
public class ExceptionShortCircuitTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(ExceptionShortCircuitTransformer.class);

	public ExceptionShortCircuitTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "EXCEPTION_SHORTCIRCUIT_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());
		List<String> scopes = getTargetScope(this.parameters.getVariableScope());

		logger.debug("Try to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// randomly combine scope
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

			// finally perform injection
			if (this.inject(b, scope)) {
				this.parameters.setInjected(true);
				this.recordInjectionInfo();
				logger.debug("Succeed to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package")
						+ " " + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
				return;
			} else {
				logger.debug("Failed injection:+ " + this.formatInjectionInfo());
			}

			logger.debug("Fail to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package") + " "
					+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
		}

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

	// for this fault type,we extract the methods with declared exceptions or with
	// try-catch blocks
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
			Chain<Trap> traps = tmpBody.getTraps();
			if ((declaredExcepts.size() == 0) && (traps.size() == 0)) {
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

	private boolean inject(Body b, String scope) {
		try {
			if (scope.equals("throw")) {

				List<SootClass> allExceptions = b.getMethod().getExceptions();
				while (true) {
					int exceptionSize = allExceptions.size();
					if (exceptionSize == 0) {
						break;
					}
					int exceptionIndex = new Random(System.currentTimeMillis()).nextInt(exceptionSize);
					SootClass targetException = allExceptions.get(exceptionIndex);
					if (this.injectThrowShort(b, targetException)) {
						return true;
					}
				}

			} else if (scope.equals("catch")) {

				List<Trap> allTraps = this.getAllTraps(b);
				while (true) {
					int trapsSize = allTraps.size();
					if (trapsSize == 0) {
						break;
					}
					int trapIndex = new Random(System.currentTimeMillis()).nextInt(trapsSize);
					Trap targetTrap = allTraps.get(trapIndex);
					allTraps.remove(trapIndex);
					if (this.injectTryShort(b, targetTrap)) {
						return true;
					}
				}
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.formatInjectionInfo());
			return false;
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

	private boolean injectTryShort(Body b, Trap trap) {
		// directly throw the exception at the beginning of the trap
		Chain<Unit> units = b.getUnits();

		Unit beginUnit = trap.getBeginUnit();
		List<Stmt> stmts = this.getShortTryStatements(b, trap);
		for (int i = 0; i < stmts.size(); i++) {
			if (i == 0) {
				// here we add the exception before the first statement if this block
				units.insertBefore(stmts.get(i), beginUnit);
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
		List<Stmt> conditionStmts = this.getConditionStmt(b, beginUnit);
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

	}

	private List<Stmt> getShortTryStatements(Body b, Trap trap) {
		// Decide the exception to be thrown
		SootClass exception = trap.getException();
		this.injectInfo.put("VariableName", exception.getName());
		// Find the constructor of this Exception
		SootMethod constructor = null;
		SootMethod constructor2 = null;
		Iterator<SootMethod> smIt = exception.getMethods().iterator();
		while (smIt.hasNext()) {
			SootMethod tmp = smIt.next();
			String signature = tmp.getSignature();
			// TO-DO
			// we should also decide which constructor should be used
			if (signature.contains("void <init>(java.lang.String)")) {
				constructor = tmp;
			}
			if (signature.contains("void <init>()")) {
				constructor2 = tmp;
			}
		}
		if ((constructor == null) && (constructor2 == null)) {
			// this is a fatal error
			logger.error("Failed to find a constructor for this exception");
			return null;
		}
		// create a exception and initialize it
		if (constructor != null) {
			Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception));
			b.getLocals().add(lexception);
			AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception)));
			InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception,
					constructor.makeRef(), StringConstant.v("Fault Injection")));
			ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);
			List<Stmt> stmts = new ArrayList<Stmt>();
			stmts.add(assignStmt);
			stmts.add(invStmt);
			stmts.add(throwStmt);
			return stmts;
		} else {
			Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception));
			b.getLocals().add(lexception);
			AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception)));
			InvokeStmt invStmt = Jimple.v()
					.newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor2.makeRef()));
			ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);
			List<Stmt> stmts = new ArrayList<Stmt>();
			stmts.add(assignStmt);
			stmts.add(invStmt);
			stmts.add(throwStmt);
			return stmts;
		}

	}

	private boolean injectThrowShort(Body b, SootClass exception) {
		// TODO Auto-generated method stub
		Chain<Unit> units = b.getUnits();

		this.injectInfo.put("VariableName", exception.getName());
		Unit firstUnit = units.getFirst();
		List<Stmt> stmts = this.getShortThrowStatements(b, exception);
		for (int i = 0; i < stmts.size(); i++) {
			if (i == 0) {
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

	}

	private List<Stmt> getShortThrowStatements(Body b, SootClass exception) {
		// Find the constructor of this Exception
		SootMethod constructor = null;
		Iterator<SootMethod> smIt = exception.getMethods().iterator();
		while (smIt.hasNext()) {
			SootMethod tmp = smIt.next();
			String signature = tmp.getSignature();
			// TO-DO
			// we should also decide which constructor should be used
			if (signature.contains("void <init>(java.lang.String)")) {
				constructor = tmp;
			}
		}

		// create a exception and initialize it
		Local tmpException = Jimple.v().newLocal("tmpSootException", RefType.v(exception));
		b.getLocals().add(tmpException);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(tmpException, Jimple.v().newNewExpr(RefType.v(exception)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(tmpException,
				constructor.makeRef(), StringConstant.v("Fault Injection")));

		// Throw it directly
		ThrowStmt throwStmt = Jimple.v().newThrowStmt(tmpException);

		List<Stmt> stmts = new ArrayList<Stmt>();
		stmts.add(assignStmt);
		stmts.add(invStmt);
		stmts.add(throwStmt);
		return stmts;
	}

	private List<String> getTargetScope(String variableScope) {
		List<String> scopes = new ArrayList<String>();
		if ((variableScope == null) || (variableScope == "")) {
			scopes.add("throw");
			scopes.add("catch");
		} else {
			scopes.add(variableScope);
		}
		return scopes;
	}

}
