package com.alex.ssfi;

import java.util.*;

import org.objectweb.asm.tree.IntInsnNode;

import soot.*;
import soot.baf.DoubleWordType;
import soot.baf.WordType;
import soot.jimple.AddExpr;
import soot.jimple.AssignStmt;
import soot.jimple.BreakpointStmt;
import soot.jimple.GotoStmt;
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
import soot.jimple.toolkits.typing.fast.BottomType;
import soot.util.*;

public class GotoTransformer extends BodyTransformer {
	private static GotoTransformer instance = new GotoTransformer();
	private boolean addedCounter=false;

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
//		SootField gotoCounter;
//		if (addedCounter) {
//			gotoCounter=Scene.v().getMainClass().getFieldByName("gotoCounter");
//		}else {
//			gotoCounter=new SootField("gotoCounter", LongType.v(), Modifier.STATIC);
//			Scene.v().getMainClass().addField(gotoCounter);
//			addedCounter=true;
//		}
////	
		
		//Scene.v().loadNecessaryClasses();
//		Scene.v().loadClassAndSupport("java.io.FileWriter");
//		SootClass fWriterClass=Scene.v().getSootClass("java.io.FileWriter");
//		Local writer=Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
//		List<SootMethod> constructor=fWriterClass.getMethods();
//		for(int i=0;i<constructor.size();i++) {
//			System.out.println(constructor.get(i).getSubSignature());
//		}
		
		String methodSignature=b.getMethod().getSignature();
		if (!methodSignature.contains("calculate")){
			return;
		}
		Chain<Unit> units=b.getUnits();
		Iterator<Unit> stmIt=units.snapshotIterator();
		while(stmIt.hasNext()) {
			Stmt stmt=(Stmt)stmIt.next();
			if (stmt instanceof SwitchStmt){
				SwitchStmt switchStmt=(SwitchStmt)stmt;
				List<Unit> targets=switchStmt.getTargets();
				for (int i=0;i<targets.size();i++) {
					System.out.println(targets.get(i).toString());
				}
				

			}else if(stmt instanceof BreakpointStmt) {
				System.out.println("break");
			}
		}
				
		
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
