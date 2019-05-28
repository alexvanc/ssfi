package com.alex.ssfi;

import java.util.Iterator;
import java.util.Map;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.LongType;
import soot.Unit;
import soot.jimple.AddExpr;
import soot.jimple.AssignStmt;
import soot.jimple.Jimple;
import soot.jimple.LongConstant;
import soot.jimple.Stmt;
import soot.util.Chain;

public class ValueTransformer extends BodyTransformer{

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
//		if (!Scene.v().getMainClass().
//		          declaresMethod("void main(java.lang.String[])"))
//		    throw new RuntimeException("couldn't find main() in mainClass");
		String methodSignature=b.getMethod().getSignature();
		if (!methodSignature.contains("calculate")){
			return;
		}
			
		System.out.println("Method signature:"+methodSignature);
		System.out.println("Phase Name:"+phaseName);
		int localNumber=b.getLocalCount();
		if (localNumber<1){
			return;
		}
		Local target=null;
		Iterator<Local> localIt=b.getLocals().iterator();
		while(localIt.hasNext()) {
			Local tmp=localIt.next();
			System.out.println("Local Type:" +tmp.getType().toString());
			if (tmp.getType() instanceof LongType) {
				target=tmp;
				break;
			}
		}
		if (target==null) {
			return;
		}
		
		System.out.println("Start to process Units");
		Chain<Unit> units=b.getUnits();
		Iterator<Unit> stmIt=units.snapshotIterator();
		while(stmIt.hasNext()) {
			Stmt stmp=(Stmt)stmIt.next();
			if (stmp instanceof AssignStmt) {
//				IdentityStmt stmp2=(IdentityStmt)stmp;
//				ValueBox lvb=stmp2.getLeftOpBox();
//				Value rv=stmp2.getRightOp();
//				System.out.println(rv.toString()+" - "+rv.getType());
//				if(rv instanceof LongType) {
//					LongType l=(LongType)rv;
//					long after=l.getNumber();
//					System.out.println("right"+after);
//					new Valu
//					lvb.setValue(value);
//				}
				
				

				
//				System.out.println("LeftOpType:"+stmp2.getLeftOp().getType().toString()+" "+stmp2.getLeftOp().toString());
				//add 1 to the local variable, we should guarantee that it is only called once and after 
				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
				Stmt incStmt=Jimple.v().newAssignStmt(target, addExp);
				units.insertAfter(incStmt, stmp);
			}
		}

	}

}
