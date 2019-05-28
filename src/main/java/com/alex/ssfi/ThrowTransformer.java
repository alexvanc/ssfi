package com.alex.ssfi;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.RefType;
import soot.SootClass;
import soot.SootMethod;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.StringConstant;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class ThrowTransformer extends BodyTransformer {
	private static final Logger logger=LogManager.getLogger(ThrowTransformer.class);

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
		String methodSignature = b.getMethod().getSignature();
		if (!methodSignature.contains("throw")) {
			return;
		}

		System.out.println("Method signature:" + methodSignature);
		System.out.println("Phase Name:" + phaseName);
		int localNumber = b.getLocalCount();
		if (localNumber < 1) {
			return;
		}

		SootMethod sm = b.getMethod();
		List<SootClass> exceptions = sm.getExceptions();

		// TO-DO
		// choose one of the exception to directly throw
		if (exceptions.size() < 1) {
			logger.info("zero exception");
			return;
		}
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
				System.out.println(tmp.getSignature());
				constructor = tmp;
			}
		}

		// create a exception and initialize it
     	Local tmpException=Jimple.v().newLocal("tmpException", RefType.v(exception));
     	b.getLocals().add(tmpException);
     	AssignStmt assignStmt = Jimple.v().newAssignStmt(tmpException, Jimple.v().newNewExpr(RefType.v(exception)));
		InvokeStmt invStmt = Jimple.v()
				.newInvokeStmt(Jimple.v().newSpecialInvokeExpr(tmpException, constructor.makeRef(),StringConstant.v("Fault Injection")));

		//Throw it directly
		ThrowStmt throwStmt = Jimple.v().newThrowStmt(tmpException);
		
		// TO-DO
		// decide where to throw this exception
		// here the local variable used in the catch block should be considered
		Chain<Unit> units=b.getUnits();
		units.insertBefore(assignStmt, units.getFirst());
		units.insertAfter(invStmt, assignStmt);
		units.insertAfter(throwStmt, invStmt);
      
	}

}
