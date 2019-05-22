package com.alex.llfi;

import java.util.Iterator;
import java.util.Map;

import soot.Body;
import soot.BodyTransformer;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.NopStmt;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.Stmt;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class ThrowTransformer extends BodyTransformer{

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
        // TODO Auto-generated method stub
        String methodSignature=b.getMethod().getSignature();
        if (!methodSignature.contains("throw")){
            return;
        }
            
        System.out.println("Method signature:"+methodSignature);
        System.out.println("Phase Name:"+phaseName);
        int localNumber=b.getLocalCount();
        if (localNumber<1){
            return;
        }

//      
        System.out.println("Start to process Units");
        Chain<Unit> units=b.getUnits();
        Iterator<Unit> stmIt=units.snapshotIterator();
        while(stmIt.hasNext()) {
            
            Stmt stmp=(Stmt)stmIt.next();
            if (stmp instanceof ThrowStmt) {
//                ThrowStmt thStmt=(ThrowStmt)stmp;
                Stmt pre1Stmt=(Stmt)units.getPredOf(stmp);
                Stmt pre2Stmt=(Stmt)units.getPredOf(pre1Stmt);
                units.remove(stmp);
                units.remove(pre1Stmt);
                units.remove(pre2Stmt);
                Stmt firstStm=(Stmt)units.getFirst();
                units.insertBefore(stmp, firstStm);
                units.insertBefore(pre1Stmt, stmp);
                units.insertBefore(pre2Stmt, pre1Stmt);
                                 
            }
//            else if(stmp instanceof AssignStmt){
//                System.out.println("In Assign");
//            }else if(stmp instanceof NopStmt){
//                System.out.println("In Nop");
//            }else if(stmp instanceof IdentityStmt){
//                System.out.println("In Identity");
//            }else if(stmp instanceof InvokeStmt){
//                System.out.println("In invoke");
//            }else if(stmp instanceof GotoStmt){
//                System.out.println("In Goto");
//            }else if(stmp instanceof IfStmt){
//                System.out.println("In if");
//            }else if((stmp instanceof ReturnStmt)||(stmp instanceof ReturnVoidStmt)){
//                System.out.println("In return");
//            }else{
//                System.out.println("In Other");
//            }
        }
//      
    }

}

