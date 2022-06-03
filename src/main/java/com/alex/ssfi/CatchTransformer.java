package com.alex.ssfi;

import java.util.Iterator;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.RefType;
import soot.SootClass;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.StringConstant;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class CatchTransformer extends BodyTransformer {
    private static final Logger logger = LogManager.getLogger(CatchTransformer.class);

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
        // TODO Auto-generated method stub
        String methodSignature = b.getMethod().getSignature();
        if (!methodSignature.contains("try")) {
            return;
        }

        logger.info("Method signature:" + methodSignature);
        logger.info("Phase Name:" + phaseName);
        int localNumber = b.getLocalCount();
        if (localNumber < 1) {
            return;
        }

//        SootMethod sm=b.getMethod();
//        SootClass scl=sm.getDeclaringClass();

        Chain<Unit> units = b.getUnits();

//        Iterator<Unit> uit=units.snapshotIterator();
//        while(uit.hasNext()) {
//        	Stmt tmpStmt=(Stmt)uit.next();
//        	Reporter.checkStmtType(tmpStmt);
//        	if(tmpStmt instanceof AssignStmt) {
//        		AssignStmt assignStmt=(AssignStmt)tmpStmt;
//        		Value lv=assignStmt.getLeftOp();
//        		Value rv=assignStmt.getRightOp();
//        		logger.info("\nWhether left Local: "+(lv instanceof Local));
//        		logger.info("\nWhether right Local: "+ (rv instanceof Local));
//        		logger.info("\nWhether right constant: "+ (rv instanceof Constant));
//        		logger.info("\nWhether right Expr: "+ (rv instanceof Expr));
//        		logger.info("\nWhether right AnyNewExpr: "+ (rv instanceof AnyNewExpr));
//        		logger.info("\nWhether right NewExpr: "+ (rv instanceof NewExpr));
//        		logger.info("\nWhether right JNewExpr: "+ (rv instanceof JNewExpr));
//        		logger.info("\nWhether right AbstractNewExpr: "+ (rv instanceof AbstractNewExpr));
//        		logger.info("\nWhether right Ref: "+ (rv instanceof Ref));
//        	}
//        }

        Iterator<Trap> traps = b.getTraps().snapshotIterator();
//        while (traps.hasNext()) {
        if (traps.hasNext()) {
            Trap tmpTrap = traps.next();
//            tmpTrap.get
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
                    System.out.println(tmp.getSignature());
                    constructor = tmp;
                }
            }
            // create a exception and initialize it
            Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception));
            b.getLocals().add(lexception);
            AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception)));
            InvokeStmt invStmt = Jimple.v()
                    .newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef(), StringConstant.v("Fault Injection")));
//            InvokeExpr invExp=Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef());
//            AssignStmt assignStmt=Jimple.v().newAssignStmt(lexception, invExp);

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

        }

    }

}
