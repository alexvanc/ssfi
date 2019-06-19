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

	private void generateTargetMethod(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		if (this.methodIndex >= allMethods.size()) {
			return;
		}
		int targetMethodIndex = new Random(System.currentTimeMillis())
				.nextInt(allMethods.size() - this.methodIndex + 1);
		// here we currently don't check whether the target method declares or catch
		// exceptions
//		SootMethod method=allMethods.get(this.methodIndex + targetMethodIndex - 1);
//		Body body=method.getActiveBody();
//		List<SootClass> exceptions=method.getExceptions();
//		Chain<Trap> traps=body.getTraps();
//		while((exceptions.size()==0)&&(traps.size()==0)){
//			targetMethodIndex = new Random(System.currentTimeMillis())
//					.nextInt(allMethods.size() - this.methodIndex + 1);
//			method=allMethods.get(this.methodIndex + targetMethodIndex - 1);
//			body=method.getActiveBody();
//			exceptions=method.getExceptions();
//			traps=body.getTraps();
//		}
		this.foundTargetMethod = true;
		this.targetMethodSubSignature = allMethods.get(this.methodIndex + targetMethodIndex - 1).getSubSignature();
		return;
	}

	private boolean inject(Body b, String scope) {
		if (scope.equals("throw")) {
			if (this.injectThrowShort(b)) {
	
				return true;
			}
		} else if (scope.equals("catch")) {
			if (this.injectTryShort(b)) {
	
				return true;
			}
		}
		return false;
	}

	private boolean injectTryShort(Body b) {
		// randomly choose a trap and directly throw the exception at the beginning of
		// the trap
		Chain<Unit> units = b.getUnits();
		Chain<Trap> traps = b.getTraps();
		if (traps.size() == 0) {
			return false;
		}
		int targetTrapIndex = new Random(System.currentTimeMillis()).nextInt(traps.size());
		Iterator<Trap> trapsItr = b.getTraps().snapshotIterator();
		int index = 0;
		try {
			while (trapsItr.hasNext()) {
				Trap tmpTrap = trapsItr.next();
				if (index == targetTrapIndex) {
					Unit beginUnit = tmpTrap.getBeginUnit();
					List<Stmt> stmts = this.getShortTryStatements(b, tmpTrap);
					for (int i = 0; i < stmts.size(); i++) {
						if (i == 0) {
							// here we add the exception before the first statement if this block
							units.insertBefore(stmts.get(i), beginUnit);
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
				} else {
					index++;
				}
			}
		} catch (Exception e) {
			logger.error(e.getMessage());
			logger.error(this.injectInfo.toString());
			return false;
		}
		return false;
	}

	private List<Stmt> getShortTryStatements(Body b, Trap trap) {
		// Decide the exception to be thrown
		SootClass exception = trap.getException();
		this.injectInfo.put("VariableName", exception.getName());
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
		Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception));
		b.getLocals().add(lexception);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef(),
				StringConstant.v("Fault Injection")));
		ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);
		List<Stmt> stmts = new ArrayList<Stmt>();
		stmts.add(assignStmt);
		stmts.add(invStmt);
		stmts.add(throwStmt);
		return stmts;
	}

	private boolean injectThrowShort(Body b) {
		// TODO Auto-generated method stub
		Chain<Unit> units=b.getUnits();
		SootMethod sm = b.getMethod();
		List<SootClass> exceptions = sm.getExceptions();
		if (exceptions.size() == 0) {
			return false;
		}
		try {
			int targetExceptionIndex = new Random(System.currentTimeMillis()).nextInt(exceptions.size());
			SootClass exception = exceptions.get(targetExceptionIndex);
			this.injectInfo.put("VariableName", exception.getName());
			List<Stmt> stmts=this.getShortThrowStatements(b, exception);
			for (int i = 0; i < stmts.size(); i++) {
				if (i == 0) {
					// currently we add the field null to the begining of the method
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
