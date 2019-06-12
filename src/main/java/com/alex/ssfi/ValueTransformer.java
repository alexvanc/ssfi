package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
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
import soot.ByteType;
import soot.Local;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.AddExpr;
import soot.jimple.AssignStmt;
import soot.jimple.DoubleConstant;
import soot.jimple.FieldRef;
import soot.jimple.FloatConstant;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.LongConstant;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.SubExpr;
import soot.util.Chain;

/*
 * TODO
 * distinguish between static field and instance field
 * report the Soot bugs of failing to get the fields of a class
 */
public class ValueTransformer extends BodyTransformer{
	private Map<String,String> injectInfo=new HashMap<String, String>();
	private RunningParameter parameters;
	private  final Logger logger=LogManager.getLogger(ValueTransformer.class);
	private  final Logger recorder=LogManager.getLogger("inject_recorder");
	
	public ValueTransformer(RunningParameter parameters) {
		this.parameters=parameters;
		
	}
	public ValueTransformer() {
	}
	

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub
//		if (!Scene.v().getMainClass().
//		          declaresMethod("void main(java.lang.String[])"))
//		    throw new RuntimeException("couldn't find main() in mainClass");
		if(this.parameters.isInjected()) {
			return;
		}
		String methodSignature=b.getMethod().getName();
		String targetMethodName=this.parameters.getMethodName();
		if((targetMethodName!=null)&&(!targetMethodName.equalsIgnoreCase(methodSignature))){
			return;
		}
		this.injectInfo.put("FaultType", "Value_FAULT");
		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
		this.injectInfo.put("Method", b.getMethod().getSubSignature());
		List<String> scopes=getTargetScope(this.parameters.getVariableScope());
		Map<String, List<Object>> allVariables=this.getAllVariables(b);
		
		//choose one scope to inject faults
		//if not specified, then randomly choose
		while(true) {
			int scopesSize=scopes.size();
			if(scopesSize==0) {
				logger.debug("Cannot find qualified scopes");
				break;
			}
			int scopeIndex=new Random().nextInt(scopesSize);
			String scope=scopes.get(scopeIndex);
			scopes.remove(scopeIndex);
			this.injectInfo.put("VariableScope", scope);
			List<String> types=getTargetType(this.parameters.getVariableType());
			while(true) {
				int typesSize=types.size();
				if(typesSize==0) {
					break;
				}
				int typeIndex=new Random().nextInt(typesSize);
				String type=types.get(typeIndex);
				types.remove(typeIndex);
				this.injectInfo.put("VariableType", type);
				if(this.inject(b, allVariables, scope, type)) {
					break;
				}
				
			}
		}

		
//		System.out.println("Start to process Units");
//		Chain<Unit> units=b.getUnits();
//		Iterator<Unit> stmIt=units.snapshotIterator();
//		while(stmIt.hasNext()) {
//			Stmt stmp=(Stmt)stmIt.next();
//			if (stmp instanceof AssignStmt) {
////				IdentityStmt stmp2=(IdentityStmt)stmp;
////				ValueBox lvb=stmp2.getLeftOpBox();
////				Value rv=stmp2.getRightOp();
////				System.out.println(rv.toString()+" - "+rv.getType());
////				if(rv instanceof LongType) {
////					LongType l=(LongType)rv;
////					long after=l.getNumber();
////					System.out.println("right"+after);
////					new Valu
////					lvb.setValue(value);
////				}
//				
//				
//
//				
////				System.out.println("LeftOpType:"+stmp2.getLeftOp().getType().toString()+" "+stmp2.getLeftOp().toString());
//				//add 1 to the local variable, we should guarantee that it is only called once and after 
//				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
//				Stmt incStmt=Jimple.v().newAssignStmt(target, addExp);
//				units.insertAfter(incStmt, stmp);
//			}
//		}

	}
	
	private boolean inject(Body b,Map<String,List<Object>> allVariables,String scope, String type) {
		if((scope.equals("local"))||(scope.equals("parameter"))){
			List<Object> source=allVariables.get(scope);
			List<Local> target=new ArrayList<Local>();
			for(int i=0;i<source.size();i++) {
				Local local=(Local) source.get(i);
				if(local.getType().toString().equals(type)) {
					String targetVariableName=this.parameters.getVariableName();
					if((targetVariableName==null)||(targetVariableName=="")||local.getName().equals(targetVariableName)){
						target.add(local);
					}
				}
			}
			while(true) {
				int variableSize=target.size();
				if(variableSize==0) {
					break;
				}
				int variableIndex=new Random().nextInt(variableSize);
				Local local=target.get(variableIndex);
				target.remove(variableIndex);
				this.injectInfo.put("VariableName", local.getName());
				List<String> actions=this.getTargetAction(type,this.parameters.getAction());
				while(true) {
					int actionSize=actions.size();
					if(actionSize<=0) {
						break;
					}
					int actionIndex=new Random().nextInt(actionSize);
					String action=actions.get(actionIndex);
					actions.remove(actionIndex);
					this.injectInfo.put("Action", action);
					if(this.injectLocalWithAction(b,local, action)) {
						//record
						this.recordInjectionInfo();
						this.parameters.setInjected(true);
						return true;
					}
				}
			}
		}else {
			List<Object> source=allVariables.get(scope);
			List<SootField> target=new ArrayList<SootField>();
			for(int i=0;i<source.size();i++) {
				SootField field=(SootField) source.get(i);
				if(field.getType().toString().equals(type)) {
					String targetVariableName=this.parameters.getVariableName();
					if((targetVariableName==null)||(targetVariableName=="")||field.getName().equals(targetVariableName)){
						target.add(field);
					}
				}
			}
			while(true) {
				int variableSize=target.size();
				if(variableSize==0) {
					break;
				}
				int variableIndex=new Random().nextInt(variableSize);
				SootField field=target.get(variableIndex);
				target.remove(variableIndex);
				this.injectInfo.put("VariableName", field.getName());
				List<String> actions=this.getTargetAction(type,this.parameters.getAction());
				while(true) {
					int actionSize=actions.size();
					if(actionSize<=0) {
						break;
					}
					int actionIndex=new Random().nextInt(actionSize);
					String action=actions.get(actionIndex);
					actions.remove(actionIndex);
					this.injectInfo.put("Action", action);
					if(this.injectFieldWithAction(b,field, action)) {
						//record
						this.recordInjectionInfo();
						this.parameters.setInjected(true);
						return true;
					}
				}
			}
		}
		
		return false;
	}
	
	/*
	 * TODO
	 */
	private void recordInjectionInfo() {
		StringBuffer sBuffer=new StringBuffer();
		sBuffer.append("ID:");
		sBuffer.append(this.parameters.getID());
		for(Map.Entry<String, String> entry:this.injectInfo.entrySet()) {
			sBuffer.append("\t");
			sBuffer.append(entry.getKey());
			sBuffer.append(":");
			sBuffer.append(entry.getValue());
		}
		System.out.println("HHHH");
		this.recorder.info(sBuffer.toString());
	}
	
	private boolean injectLocalWithAction(Body b,Local local,String action) {
		//inject at the beginning or inject after assignment make a lot of difference
		//currently we inject the fault after it is visited once
		Chain<Unit> units=b.getUnits();
		Iterator<Unit> unitIt=units.snapshotIterator();
		while(unitIt.hasNext()) {
			Unit tmp=unitIt.next();
			Iterator<ValueBox> valueBoxes=tmp.getUseAndDefBoxes().iterator();
			while(valueBoxes.hasNext()) {
				Value value=valueBoxes.next().getValue();
				if((value instanceof Local)&&(value.equivTo(local))) {
					logger.info(tmp.toString());
					try {
						List<Stmt> stmts=this.getLocalStatementsByAction(local, action);
						for(int i=0;i<stmts.size();i++) {
							if(i==0) {
								units.insertAfter(stmts.get(i), tmp);
							}else {
								units.insertAfter(stmts.get(i), stmts.get(i-1));
							}
						}
						
						List<Stmt> actStmts=this.createActivateStatement(b);
						for(int i=0;i<actStmts.size();i++) {
							if(i==0) {
								units.insertAfter(actStmts.get(i), stmts.get(stmts.size()-1));
							}else {
								units.insertAfter(actStmts.get(i), actStmts.get(i-1));
							}
						}
						return true;
					}catch (Exception e) {
						logger.error(e.getMessage());
						return false;
					}

				}
			}

		}
		return false;
	}
	
	private List<Stmt> getLocalStatementsByAction(Local local,String action) {
		Stmt valueChangeStmt=null;
		if(action.equals("TO")) {
			String typeName=local.getType().toString();
			String targetStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", targetStringValue);
			if(typeName.equals("byte")) {
				byte result=Byte.parseByte(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(local, IntConstant.v(result));
			}else if(typeName.equals("short")) {
				short result=Short.parseShort(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(local, IntConstant.v(result));
			}else if(typeName.equals("int")) {
				int result=Integer.parseInt(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(local, IntConstant.v(result));
			}else if(typeName.equals("long")) {
				long result=Long.parseLong(targetStringValue);
//				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
				valueChangeStmt=Jimple.v().newAssignStmt(local, LongConstant.v(result));
			}else if(typeName.equals("float")) {
				float result=Float.parseFloat(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(local, FloatConstant.v(result));
			}else if(typeName.equals("double")) {
				double result=Double.parseDouble(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(local, DoubleConstant.v(result));
//			}else if(typeName.equals("bool")) {
//				boolean result=Boolean.parseBoolean(targetStringValue);
////				valueChangeStmt=Jimple.v().newAssignStmt(local, Boolean.v(result));
//				Jimple.v().new
			}else if(typeName.equals("java.lang.String")) {
				valueChangeStmt=Jimple.v().newAssignStmt(local, StringConstant.v(targetStringValue));
			}			
		}else if(action.equals("ADD")){
			String typeName=local.getType().toString();
			String addedStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", addedStringValue);
			if(typeName.equals("byte")) {
				byte addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Byte.parseByte(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
			}else if(typeName.equals("short")) {
				short addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Short.parseShort(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("int")) {
				int addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Integer.parseInt(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("long")) {
				long addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Long.parseLong(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, LongConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("float")) {
				float addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Float.parseFloat(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
			}else if(typeName.equals("double")) {
				double addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Double.parseDouble(addedStringValue);
				}
				AddExpr addExp=Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
//			}else if(typeName.equals("bool")) {
//				
			}
			
		}else if(action.equals("SUB")) {
			String typeName=local.getType().toString();
			String subedStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", subedStringValue);
			if(typeName.equals("byte")) {
				byte addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Byte.parseByte(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
			}else if(typeName.equals("short")) {
				short addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Short.parseShort(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("int")) {
				int addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Integer.parseInt(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("long")) {
				long addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Long.parseLong(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, LongConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				
			}else if(typeName.equals("float")) {
				float addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Float.parseFloat(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
			}else if(typeName.equals("double")) {
				double addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Double.parseDouble(subedStringValue);
				}
				SubExpr addExp=Jimple.v().newSubExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
//			}else if(typeName.equals("bool")) {
//				
			}
		}
		List<Stmt> stmts=new ArrayList<Stmt>();
		stmts.add(valueChangeStmt);
		return stmts;
	}
	
	
	private List<Stmt> getFieldStatementsByAction(Body b,SootField field,String action) {
		//TODO
		//verify the filed API bugs
		Stmt valueChangeStmt=null;
		Stmt copyFieldStmt=null;
		Stmt changeFieldStmt=null;
		List<Stmt> stmts=new ArrayList<Stmt>();
		Local local=null;
		if(action.equals("TO")) {
			String typeName=field.getType().toString();
			String targetStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", targetStringValue);
			if(typeName.equals("byte")) {
//				
				byte result=Byte.parseByte(targetStringValue);
//				Jimple.v().newSt
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()),IntConstant.v(result));

//				valueChangeStmt=Jimple.v().newAssignStmt(field., IntConstant.v(result));
			}else if(typeName.equals("short")) {
				short result=Short.parseShort(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()),IntConstant.v(result));
			}else if(typeName.equals("int")) {
				int result=Integer.parseInt(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()),IntConstant.v(result));
			}else if(typeName.equals("long")) {
				long result=Long.parseLong(targetStringValue);
//				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()),LongConstant.v(result));
			}else if(typeName.equals("float")) {
				float result=Float.parseFloat(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()), FloatConstant.v(result));
			}else if(typeName.equals("double")) {
				double result=Double.parseDouble(targetStringValue);
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()), DoubleConstant.v(result));
//			}else if(typeName.equals("bool")) {
//				boolean result=Boolean.parseBoolean(targetStringValue);
////				valueChangeStmt=Jimple.v().newAssignStmt(local, Boolean.v(result));
//				Jimple.v().new
			}else if(typeName.equals("java.lang.String")) {
				valueChangeStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()), StringConstant.v(targetStringValue));
			}		
			stmts.add(valueChangeStmt);
		}else if(action.equals("ADD")){
			String typeName=field.getType().toString();
			String addedStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", addedStringValue);
			if(typeName.equals("byte")) {
				byte addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Byte.parseByte(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
			}else if(typeName.equals("short")) {
				short addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Short.parseShort(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("int")) {
				int addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Integer.parseInt(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("long")) {
				long addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Long.parseLong(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, LongConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("float")) {
				float addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Float.parseFloat(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
			}else if(typeName.equals("double")) {
				double addedValue=1;
				if((addedStringValue!=null)&&(addedStringValue!="")) {
					addedValue=Double.parseDouble(addedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp=Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
//			}else if(typeName.equals("bool")) {
//				
			}
			stmts.add(copyFieldStmt);
			stmts.add(valueChangeStmt);
			stmts.add(changeFieldStmt);
		}else if(action.equals("SUB")) {
			String typeName=field.getType().toString();
			String subedStringValue=this.parameters.getVariableValue();
			this.injectInfo.put("VariableValue", subedStringValue);
			if(typeName.equals("byte")) {
				byte addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Byte.parseByte(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
			}else if(typeName.equals("short")) {
				short addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Short.parseShort(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("int")) {
				int addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Integer.parseInt(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("long")) {
				long addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Long.parseLong(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, LongConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				
			}else if(typeName.equals("float")) {
				float addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Float.parseFloat(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
			}else if(typeName.equals("double")) {
				double addedValue=1;
				if((subedStringValue!=null)&&(subedStringValue!="")) {
					addedValue=Double.parseDouble(subedStringValue);
				}
				local=Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp=Jimple.v().newSubExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt=Jimple.v().newAssignStmt(local, addExp);
				changeFieldStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
//			}else if(typeName.equals("bool")) {
			}
			stmts.add(copyFieldStmt);
			stmts.add(valueChangeStmt);
			stmts.add(changeFieldStmt);
		}
		return stmts;
	}
	/*
	 * Print this activation information to a file
	 */
	private List<Stmt> createActivateStatement(Body b) {
		SootClass fWriterClass=Scene.v().getSootClass("java.io.FileWriter");
		Local writer=Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
		b.getLocals().add(writer);
		SootMethod constructor=fWriterClass.getMethod("void <init>(java.lang.String,boolean)");
		SootMethod printMethod=Scene.v().getMethod("<java.io.Writer: void write(java.lang.String)>");
		SootMethod closeMethod=Scene.v().getMethod("<java.io.OutputStreamWriter: void close()>");
		
		AssignStmt newStmt=Jimple.v().newAssignStmt(writer, Jimple.v().newNewExpr(RefType.v("java.io.FileWriter")));
		InvokeStmt invStmt = Jimple.v()
				.newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(), StringConstant.v(this.parameters.getOutput()+File.separator+"activation.txt"),IntConstant.v(1)));
		InvokeStmt logStmt=Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(), StringConstant.v(this.parameters.getID()+"\n")));
		InvokeStmt closeStmt=Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, closeMethod.makeRef()));

		List<Stmt> statements=new ArrayList<Stmt>();
		statements.add(newStmt);
		statements.add(invStmt);
		statements.add(logStmt);
		statements.add(closeStmt);
		return statements;
	}
	
	private boolean injectFieldWithAction(Body b,SootField field,String action) {
		Chain<Unit> units=b.getUnits();
		Iterator<Unit> unitIt=units.snapshotIterator();
		while(unitIt.hasNext()) {
			Unit tmp=unitIt.next();
			Iterator<ValueBox> valueBoxes=tmp.getUseAndDefBoxes().iterator();
			while(valueBoxes.hasNext()) {
				Value value=valueBoxes.next().getValue();
				//TODO
				//check whether SootField in Soot is in SingleTon mode for equals comparison
				if((value instanceof FieldRef)&&(((FieldRef)value).getField().equals(field))) {
					logger.info(tmp.toString());
					try {
						List<Stmt> stmts=this.getFieldStatementsByAction(b,field, action);
						for(int i=0;i<stmts.size();i++) {
							if(i==0) {
								units.insertAfter(stmts.get(i), tmp);
							}else {
								units.insertAfter(stmts.get(i), stmts.get(i-1));
							}
						}
						
						List<Stmt> actStmts=this.createActivateStatement(b);
						for(int i=0;i<actStmts.size();i++) {
							if(i==0) {
								units.insertAfter(actStmts.get(i), stmts.get(stmts.size()-1));
							}else {
								units.insertAfter(actStmts.get(i), actStmts.get(i-1));
							}
						}
						return true;
					}catch (Exception e) {
						logger.error(e.getMessage());
						return false;
					}

				}
			}

		}
		return false;
	}
	
	private Map<String,List<Object>> getAllVariables(Body b){
		Map<String,List<Object>> result=new HashMap<String, List<Object>>();
		List<Local> tmpPLocals=b.getParameterLocals();
		List<Object> pLocals=new ArrayList<Object>();
		Chain<Local> tmpLocals=b.getLocals();
		List<Object> locals=new ArrayList<Object>();
		Chain<SootField> tmpFields=b.getMethod().getDeclaringClass().getFields();
		List<Object> fields=new ArrayList<Object>();
		
		int i;
		for(i=0;i<tmpPLocals.size();i++) {
			Local local=tmpPLocals.get(i);
			if(this.isTargetedType(local.getType().toString())&&(!local.getName().startsWith("$"))) {
				pLocals.add(local);
			}
		}
		
		Iterator<Local> tmpLocalsItr=tmpLocals.iterator();
		while(tmpLocalsItr.hasNext()) {
			Local local=tmpLocalsItr.next();
			if(!pLocals.contains(local)) {
				if(this.isTargetedType(local.getType().toString())&&(!local.getName().startsWith("$"))) {
					locals.add(local);
				}
			}
			
		}
		
		//actually soot cannot get Class Fields
		Iterator<SootField> tmpFieldItr=tmpFields.iterator();
		while(tmpFieldItr.hasNext()) {
			SootField field=tmpFieldItr.next();
			logger.info("Field name:　"+field.getName()+" Field type: "+field.getType().toString());
			if(this.isTargetedType(field.getType().toString())&&(!field.getName().startsWith("$"))) {
				fields.add(field);
			}
		}
		result.put("parameter", pLocals);
		result.put("local", locals);
		result.put("field", fields);
		return result;
	}
//	private boolean performInjection(String scope,String type,String name,Body b) {
//		b.getP
//		if(scope.equals("field")) {
//			b.getMethod().getDeclaringClass().get
//		}
//		return false;
//	}
	private List<String> getTargetScope(String variableScope){
		List<String> scopes=new ArrayList<String>();
		if ((variableScope==null) ||(variableScope=="")){
			scopes.add("local");
			scopes.add("field");
			scopes.add("parameter");
		}else {
			scopes.add(variableScope);
		}
		return scopes;
	}
	
	private List<String> getTargetAction(String type,String action){
		List<String> actions=new ArrayList<String>();
		if((action!=null)&&(action!="")) {
			actions.add(action);
			return actions;
		}
		if(type.equals("java.lang.String")) {
			actions.add("TO");
		}else {
			actions.add("ADD");
			actions.add("SUB");
			actions.add("TO");
			
		}
		return actions;
	}
	
	private List<String> getTargetType(String variableType){
		List<String> types=new ArrayList<String>();
		if((variableType==null)||(variableType=="")) {
			types.add("byte");
			types.add("short");
			types.add("int");
			types.add("long");
			types.add("float");
			types.add("double");
			types.add("string");
//			types.add("bool");
		}else {
			types.add(variableType);
		}
		return types;
	}
	
	private boolean isTargetedType(String typeName) {
//		String[] targetTypes= {"boolean","byte","short","int","long","float","double","java.lang.String"};
		String[] targetTypes= {"byte","short","int","long","float","double","java.lang.String"};
		List<String> list=Arrays.asList(targetTypes);
		if(list.contains(typeName)) {
			return true;
		}else {
			return false;
		}
	}

}

