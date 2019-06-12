package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.IntConstant;
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
public class ExceptionShortCircuitTransformer extends BodyTransformer {
	private Map<String, String> injectInfo = new HashMap<String, String>();
	private RunningParameter parameters;
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);
	private final Logger recorder = LogManager.getLogger("inject_recorder");

	public ExceptionShortCircuitTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
		if (this.parameters.isInjected()) {
			return;
		}
		String methodSignature = b.getMethod().getName();
		String targetMethodName = this.parameters.getMethodName();
		if ((targetMethodName != null) && (!targetMethodName.equalsIgnoreCase(methodSignature))) {
			return;
		}
		this.injectInfo.put("FaultType", "EXCEPTION_SHORTCIRCUIT_FAULT");
		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
		List<String> scopes = getTargetScope(this.parameters.getVariableScope());

		// choose one scope to inject faults
		// if not specified, then randomly choose
		while (true) {
			int scopesSize = scopes.size();
			if (scopesSize == 0) {
				logger.debug("Cannot find qualified scopes");
				break;
			}
			int scopeIndex = new Random().nextInt(scopesSize);
			String scope = scopes.get(scopeIndex);
			scopes.remove(scopeIndex);

			if (this.inject(b, scope)) {
				break;
			}
		}

	}

	private List<Stmt> createActivateStatement(Body b) {
		SootClass fWriterClass = Scene.v().getSootClass("java.io.FileWriter");
		Local writer = Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
		b.getLocals().add(writer);
		SootMethod constructor = fWriterClass.getMethod("void <init>(java.lang.String,boolean)");
		SootMethod printMethod = Scene.v().getMethod("<java.io.Writer: void write(java.lang.String)>");
		SootMethod closeMethod = Scene.v().getMethod("<java.io.OutputStreamWriter: void close()>");

		AssignStmt newStmt = Jimple.v().newAssignStmt(writer, Jimple.v().newNewExpr(RefType.v("java.io.FileWriter")));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
				StringConstant.v(this.parameters.getOutput() + File.separator + "activation.txt"), IntConstant.v(1)));
		InvokeStmt logStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(),
				StringConstant.v(this.parameters.getID() + "\n")));
		InvokeStmt closeStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, closeMethod.makeRef()));

		List<Stmt> statements = new ArrayList<Stmt>();
		statements.add(newStmt);
		statements.add(invStmt);
		statements.add(logStmt);
		statements.add(closeStmt);
		return statements;
	}

	private boolean inject(Body b, String scope) {
		try {
			Chain<Unit> units = b.getUnits();
			if (scope.equals("throw")) {
				List<SootClass> exceptions = b.getMethod().getExceptions();
				if ((exceptions == null) || (exceptions.size() == 0)) {
					// this method doesn't declare any throw exceptions
					return false;
				}
				this.injectInfo.put("Method", b.getMethod().getSubSignature());
				this.injectInfo.put("VariableScope", scope);
				if (this.injectThrowShort(b)) {
					List<Stmt> actStmts = this.createActivateStatement(b);
					for (int i = 0; i < actStmts.size(); i++) {
						if (i == 0) {
							units.insertBefore(actStmts.get(i), units.getFirst());
						} else {
							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
						}
					}
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
					return true;
				}

			} else {// catch exception
				Chain<Trap> traps = b.getTraps();
				if ((traps == null) || (traps.isEmpty())) {
					// this method doesn't have have try/catch blocks
				}
				if (this.injectTryShort(b)) {
					List<Stmt> actStmts = this.createActivateStatement(b);
					for (int i = 0; i < actStmts.size(); i++) {
						if (i == 0) {
							units.insertBefore(actStmts.get(i), units.getFirst());
						} else {
							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
						}
					}
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
					return true;
				}
			}

		} catch (Exception e) {
			logger.error(e.getMessage());
			return false;
		}
		return false;
	}

	private void recordInjectionInfo() {
		StringBuffer sBuffer = new StringBuffer();
		sBuffer.append("ID:");
		sBuffer.append(this.parameters.getID());
		for (Map.Entry<String, String> entry : this.injectInfo.entrySet()) {
			sBuffer.append("\t");
			sBuffer.append(entry.getKey());
			sBuffer.append(":");
			sBuffer.append(entry.getValue());
		}
		this.recorder.info(sBuffer.toString());
	}

	private boolean injectTryShort(Body b) {
		// TODO Auto-generated method stub
		Chain<Unit> units = b.getUnits();

		Iterator<Trap> traps = b.getTraps().snapshotIterator();
//      while (traps.hasNext()) {
		if (traps.hasNext()) {
			Trap tmpTrap = traps.next();
//          tmpTrap.get
			Unit sUnit = tmpTrap.getBeginUnit();

			// Decide the exception to be thrown
			SootClass exception = tmpTrap.getException();
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
			InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception,
					constructor.makeRef(), StringConstant.v("Fault Injection")));
//          InvokeExpr invExp=Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef());
//          AssignStmt assignStmt=Jimple.v().newAssignStmt(lexception, invExp);

			ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);
//			GotoStmt gotoStmt=Jimple.v().newGotoStmt(tUnit);

			// TO-DO
			// decide where to throw this exception
			// here the local variable used in the catch block should be considered
			units.insertBefore(assignStmt, sUnit);
			units.insertBefore(invStmt, sUnit);
			units.insertBefore(throwStmt, sUnit);
//			units.insertBefore(gotoStmt, sUnit);
			tmpTrap.setBeginUnit(assignStmt);
			return true;
		} else {
			return false;
		}

	}

	private boolean injectThrowShort(Body b) {
		// TODO Auto-generated method stub
		SootMethod sm = b.getMethod();
		List<SootClass> exceptions = sm.getExceptions();
		SootClass exception = exceptions.get(0);
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
		Local tmpException = Jimple.v().newLocal("tmpException", RefType.v(exception));
		b.getLocals().add(tmpException);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(tmpException, Jimple.v().newNewExpr(RefType.v(exception)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(tmpException,
				constructor.makeRef(), StringConstant.v("Fault Injection")));

		// Throw it directly
		ThrowStmt throwStmt = Jimple.v().newThrowStmt(tmpException);

		// TO-DO
		// decide where to throw this exception
		// here the local variable used in the catch block should be considered
		Chain<Unit> units = b.getUnits();
		units.insertBefore(assignStmt, units.getFirst());
		units.insertAfter(invStmt, assignStmt);
		units.insertAfter(throwStmt, invStmt);
		return true;

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
