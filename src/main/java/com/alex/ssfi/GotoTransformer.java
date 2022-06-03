package com.alex.ssfi;

import java.util.*;

import org.objectweb.asm.tree.IntInsnNode;

import soot.*;
import soot.baf.DoubleWordType;
import soot.baf.WordType;
import soot.baf.internal.BafLocalBox;
import soot.grimp.internal.ExprBox;
import soot.grimp.internal.GRValueBox;
import soot.grimp.internal.ObjExprBox;
import soot.jimple.AddExpr;
import soot.jimple.AssignStmt;
import soot.jimple.BreakpointStmt;
import soot.jimple.EqExpr;
import soot.jimple.FieldRef;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeExpr;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.LongConstant;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.StaticInvokeExpr;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.SwitchStmt;
import soot.jimple.internal.ConditionExprBox;
import soot.jimple.internal.IdentityRefBox;
import soot.jimple.internal.ImmediateBox;
import soot.jimple.internal.InvokeExprBox;
import soot.jimple.internal.JimpleLocalBox;
import soot.jimple.internal.RValueBox;
import soot.jimple.internal.VariableBox;
import soot.jimple.toolkits.typing.fast.BottomType;
import soot.util.*;

public class GotoTransformer extends BodyTransformer {
    private static final GotoTransformer instance = new GotoTransformer();
    private final boolean addedCounter = false;

    private GotoTransformer() {
    }

    public static GotoTransformer v() {
        return instance;
    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
        // TODO Auto-generated method stub
//		if (!Scene.v().getMainClass().
//		          declaresMethod("void main(java.lang.String[])"))
//		    throw new RuntimeException("couldn't find main() in mainClass");
//		boolean isMainMethod=b.getMethod().isMain();
////		SootMethod thisMethod=b.getMethod().get;


////	
//		Chain<SootField> allFields=b.getMethod().getDeclaringClass().getFields();
//		System.out.println(allFields.size());

        //Scene.v().loadNecessaryClasses();
//		Scene.v().loadClassAndSupport("java.io.FileWriter");
//		SootClass fWriterClass=Scene.v().getSootClass("java.io.FileWriter");
//		Local writer=Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
//		List<SootMethod> constructor=fWriterClass.getMethods();
//		for(int i=0;i<constructor.size();i++) {
//			System.out.println(constructor.get(i).getSubSignature());
//		}
//		System.out.println("In Method: "+b.getMethod().getName());
//		List<SootMethod> methods=b.getMethod().getDeclaringClass().getMethods();
//		for(int i=0;i<methods.size();i++) {
//			Body body=methods.get(i).getActiveBody();
//			int number=body.getUnits().size();
//			
//			System.out.println("Body name: "+body.getMethod().getName()+" units nubmer: "+number);
//			if(body.getMethod().getName().equals("main")) {
//				Iterator<Unit> units=body.getUnits().snapshotIterator();
//				while(units.hasNext()) {
//					Unit unit=units.next();
//					System.out.println(unit.toString());
//				}
////				System.out.println(options.toString());
//			}
//		}

        String methodSignature = b.getMethod().getSignature();
        if (!methodSignature.contains("tryAndCatch")) {
            return;
        }
        Iterator<Trap> trapItr = b.getTraps().snapshotIterator();
        while (trapItr.hasNext()) {
            Trap tmp = trapItr.next();
            System.out.println(tmp.getBeginUnit().toString());
            System.out.println(tmp.getEndUnit().toString());
            System.out.println(tmp.getHandlerUnit().toString());
        }
//		
//		SootField gotoCounter;
//		try {
//			gotoCounter=Scene.v().getMainClass().getField("sootInjected",BooleanType.v());
//		}catch(Exception e) {
//			gotoCounter=new SootField("gotoCounter", LongType.v(), Modifier.STATIC);
//			b.getMethod().getDeclaringClass().addField(gotoCounter);
//		}
//		
//		Chain<Unit> units=b.getUnits();
//		Iterator<Unit> stmIt=units.snapshotIterator();
//		while(stmIt.hasNext()) {
//			Stmt stmt=(Stmt)stmIt.next();
//			if(stmt instanceof InvokeStmt) {
//				EqExpr compareExp=Jimple.v().newEqExpr(Jimple.v().newStaticFieldRef(gotoCounter.makeRef()), IntConstant.v(1));
//				IfStmt checkCondition=Jimple.v().newIfStmt(Jimple.v().newConditionExprBox(compareExp).getValue(), stmt);
//				AssignStmt injectFlagStmt=Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(gotoCounter.makeRef()), IntConstant.v(1));
//				GotoStmt afterElseStmt=Jimple.v().newGotoStmt(units.getSuccOf(stmt));
//				units.insertBefore(checkCondition, stmt);
//				units.insertAfter(injectFlagStmt, checkCondition);
//				units.insertAfter(afterElseStmt, injectFlagStmt);
//			}
//			
//			
//
//		}


//		List<SootMethod> methods=b.getMethod().getDeclaringClass().getMethods();
//		for(int i=0;i<methods.size();i++) {
//			Body body=methods.get(i).getActiveBody();
//			Chain<Trap> trapChain=b.getTraps();
//			Iterator<Trap> trapItr=trapChain.snapshotIterator();
//			while(trapItr.hasNext()) {
//				Trap trap=trapItr.next();
//				System.out.println(trap.getEndUnit().toString());
//				System.out.println(trap.getHandlerUnit().toString());
//			}
//			
//		}
//		List<SootClass> exceptions=b.getMethod().getExceptions();
//		System.out.println(exceptions.size());
//		Iterator<Local> locals=b.getLocals().snapshotIterator();
//		while(locals.hasNext()) {
//			Local tmp=locals.next();
//			System.out.println("tmpName: "+tmp.getName()+" tmpType: "+tmp.getType().toString());
//			if (tmp.getName().equals("flag")){
//	
//				System.out.println("RefLikeType" +(tmp.getType() instanceof RefLikeType));
//				System.out.println("AnyAubType" +(tmp.getType() instanceof AnySubType));
//				System.out.println("ArrayType" +(tmp.getType() instanceof ArrayType));
//				System.out.println("NullType" +(tmp.getType() instanceof NullType));
//				System.out.println("RefType" +(tmp.getType() instanceof RefType));
//				
//			}
//		}

//		Chain<Unit> units=b.getUnits();
//		Iterator<Unit> stmIt=units.snapshotIterator();
//		while(stmIt.hasNext()) {
//			Stmt stmt=(Stmt)stmIt.next();
//			if (stmt instanceof GotoStmt){
//				Local tmpLocal=Jimple.v().newLocal("tmpSoot", LongType.v());
//				b.getLocals().add(tmpLocal);
//				AssignStmt getValue=Jimple.v().newAssignStmt(tmpLocal, Jimple.v().newStaticFieldRef(gotoCounter.makeRef()));
//				units.insertBefore(getValue, stmt);
//				AddExpr addExp=Jimple.v().newAddExpr(tmpLocal, LongConstant.v(1));
//				Stmt incStmt=Jimple.v().newInvokeStmt(addExp);
//				units.insertAfter(incStmt, getValue);
//				AssignStmt addValue = Jimple.v().newAssignStmt(tmpLocal,
//                        Jimple.v().newStaticFieldRef(gotoCounter.makeRef()));
//				units.insertAfter(addValue, incStmt);
//				
//			}else if(stmt instanceof InvokeStmt) {
//				InvokeExpr iexpr = (InvokeExpr) ((InvokeStmt)stmt).getInvokeExpr();
//				if (iexpr instanceof StaticInvokeExpr)
//				{
//				    SootMethod target = ((StaticInvokeExpr)iexpr).getMethod();
//				    //if (target.) check whether isFinal means the same
//				    if (target.getSignature().equals
//				                ("<java.lang.System: void exit(int)>"))
//				    {
//						Local output=Jimple.v().newLocal("output", RefType.v("java.io.PrintStream"));;
//						b.getLocals().add(output);
//
//				    	SootMethod toCall = Scene.v().getMethod
//				    		      ("<java.io.PrintStream: void println(java.lang.Long)>");
//				    	Local tmpLocal2=Jimple.v().newLocal("tmpLocal2", LongType.v());
//						b.getLocals().add(tmpLocal2);
//						AssignStmt getValue=Jimple.v().newAssignStmt(tmpLocal2, Jimple.v().newStaticFieldRef(gotoCounter.makeRef()));
//						units.insertBefore(getValue, stmt);
//						units.insertAfter(Jimple.v().newInvokeStmt
//				    	        (Jimple.v().newVirtualInvokeExpr
//						    	           (output, toCall.makeRef(), tmpLocal2)), getValue);
//				    }
//				}
//				
//			}else if(stmt instanceof ReturnStmt || stmt instanceof ReturnVoidStmt) {
//				Local output=Jimple.v().newLocal("output", RefType.v("java.io.PrintStream"));;
//				b.getLocals().add(output);
//				//if the return is not in the main method, it main conflict with the tmpLocal2 created before exist
//
//		    	SootMethod toCall = Scene.v().getMethod
//		    		      ("<java.io.PrintStream: void println(java.lang.Long)>");
//		    	Local tmpLocal2=Jimple.v().newLocal("tmpLocal2", LongType.v());
//				b.getLocals().add(tmpLocal2);
//				AssignStmt getValue=Jimple.v().newAssignStmt(tmpLocal2, Jimple.v().newStaticFieldRef(gotoCounter.makeRef()));
//				units.insertBefore(getValue, stmt);
//				units.insertAfter(Jimple.v().newInvokeStmt
//		    	        (Jimple.v().newVirtualInvokeExpr
//				    	           (output, toCall.makeRef(), tmpLocal2)), getValue);
//				
//			}else {
//				
//			}
//			
//		}

    }

}
