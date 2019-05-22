package com.alex.llfi;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.RefType;
import soot.SootClass;
import soot.SootMethod;
import soot.Trap;
import soot.Type;
import soot.Unit;
import soot.UnitBox;
import soot.Value;
import soot.jimple.AssignStmt;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.NopStmt;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.Stmt;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class CatchTransformer extends BodyTransformer {

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
        // TODO Auto-generated method stub
        String methodSignature = b.getMethod().getSignature();
        if (!methodSignature.contains("try")) {
            return;
        }

        System.out.println("Method signature:" + methodSignature);
        System.out.println("Phase Name:" + phaseName);
        int localNumber = b.getLocalCount();
        if (localNumber < 1) {
            return;
        }

        Chain<Unit> units = b.getUnits();

        Iterator<Trap> traps = b.getTraps().snapshotIterator();
        while (traps.hasNext()) {
            Trap tmpTrap = traps.next();
            Unit eUnit = tmpTrap.getEndUnit();

            // Decide the exception to be thrown
            SootClass exception = tmpTrap.getException();
            // System.out.println(exception.getName());

            SootMethod constructor = null;
            Iterator<SootMethod> smIt = exception.getMethods().iterator();
            while (smIt.hasNext()) {
                SootMethod tmp = smIt.next();
                String signature = tmp.getSignature();
                if (signature.contains("void <init>()")) {
                    System.out.println(tmp.getSignature());
                    constructor = tmp;
                }
            }
            Unit hUnit = tmpTrap.getHandlerUnit();
            GotoStmt gotoStmt = Jimple.v().newGotoStmt(hUnit);
            // Directly go to the catch block, maybe should delete the last unit
            units.insertBefore(gotoStmt, eUnit);
//            units.remove(eUnit);
            
            Local lexception=null;
            Stmt hStmt = (Stmt) hUnit;
            if (hStmt instanceof IdentityStmt) {
                IdentityStmt idStmt=(IdentityStmt)hStmt;
                Value left=idStmt.getLeftOp();
                if (left instanceof Local){
                    lexception=(Local)left;
                    System.out.println("found");
                }
                
                Stmt nextStmt=(Stmt) units.getSuccOf(hStmt);
                
                units.remove(hStmt);
                //initialize a new Exception
                InvokeStmt invokeStmt = Jimple.v().newInvokeStmt(
                        Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef()));
                units.insertBefore(invokeStmt, nextStmt);
            }

            
//            units.insertAfter(s, invokeStmt);
            // Unit hUnit=tmpTrap.getHandlerUnit();
            // GotoStmt gotoStmp=Jimple.v().newGotoStmt(hUnit);
            // units.insertAfter(gotoStmp, bUnit);
        }

        //
        // System.out.println("Start to process Units");
        // Chain<Unit> units = b.getUnits();
        // Iterator<Unit> stmIt = units.snapshotIterator();
        // while (stmIt.hasNext()) {
        //
        // Stmt stmp = (Stmt) stmIt.next();
        // Maybe this condition is not enough
        // if ((stmp instanceof
        // IdentityStmt)&&(stmp.getBoxesPointingToThis().size()>0)) {
        // units.getSuccOf(point)

        // IdentityStmt idStmt=(IdentityStmt)stmp;
        // Type rType=idStmt.getRightOp().getType();
        // System.out.println("Type:"+rType.toString());
        // if (rType instanceof Throwable){
        // }

        // } else if (stmp instanceof AssignStmt) {
        // System.out.println("In Assign");
        // } else if (stmp instanceof NopStmt) {
        // System.out.println("In Nop");
        // } else if (stmp instanceof ThrowStmt) {
        // System.out.println("In throw");
        // } else if (stmp instanceof InvokeStmt) {
        // System.out.println("In invoke");
        // } else if (stmp instanceof GotoStmt) {
        // System.out.println("In Goto");
        // } else if (stmp instanceof IfStmt) {
        // System.out.println("In if");
        // } else if ((stmp instanceof ReturnStmt) || (stmp instanceof
        // ReturnVoidStmt)) {
        // System.out.println("In return");
        // } else {
        // System.out.println("In Other");
        // }
        // }

    }

}

