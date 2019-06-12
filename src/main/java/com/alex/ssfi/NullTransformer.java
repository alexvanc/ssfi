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
import soot.Local;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.AssignStmt;
import soot.jimple.FieldRef;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.NullConstant;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.util.Chain;

/*
 * TODO
 * distinguish between static field and instance field
 * decide whether to inject NullFault to the return value
 */
public class NullTransformer extends BodyTransformer {
	private Map<String, String> injectInfo = new HashMap<String, String>();
	private RunningParameter parameters;
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);
	private final Logger recorder = LogManager.getLogger("inject_recorder");

	public NullTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "NULL_FAULT");
		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
		this.injectInfo.put("Method", b.getMethod().getSubSignature());
		List<String> scopes = getTargetScope(this.parameters.getVariableScope());
		Map<String, List<Object>> allVariables = this.getAllVariables(b);

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
			this.injectInfo.put("VariableScope", scope);
			while (true) {
				if (this.inject(b, allVariables, scope)) {
					break;
				}

			}

		}
	}

	private boolean inject(Body b, Map<String, List<Object>> allVariables, String scope) {
		if ((scope.equals("local")) || (scope.equals("parameter"))) {
			List<Object> source = allVariables.get(scope);
			List<Local> target = new ArrayList<Local>();
			for (int i = 0; i < source.size(); i++) {
				Local local = (Local) source.get(i);
				String targetVariableName = this.parameters.getVariableName();
				if ((targetVariableName == null) || (targetVariableName == "")
						|| local.getName().equals(targetVariableName)) {
					target.add(local);
				}
			}
			while (true) {
				int variableSize = target.size();
				if (variableSize == 0) {
					break;
				}
				int variableIndex = new Random().nextInt(variableSize);
				Local local = target.get(variableIndex);
				target.remove(variableIndex);
				this.injectInfo.put("VariableName", local.getName());

				if (this.injectLocalWithNull(b, local)) {
					// record
					this.recordInjectionInfo();
					this.parameters.setInjected(true);
					return true;
				}

			}
		} else {
			List<Object> source = allVariables.get(scope);
			List<SootField> target = new ArrayList<SootField>();
			for (int i = 0; i < source.size(); i++) {
				SootField field = (SootField) source.get(i);
				String targetVariableName = this.parameters.getVariableName();
				if ((targetVariableName == null) || (targetVariableName == "")
						|| field.getName().equals(targetVariableName)) {
					target.add(field);
				}
			}
			while (true) {
				int variableSize = target.size();
				if (variableSize == 0) {
					break;
				}
				int variableIndex = new Random().nextInt(variableSize);
				SootField field = target.get(variableIndex);
				target.remove(variableIndex);
				this.injectInfo.put("VariableName", field.getName());
				if (this.injectFieldWithNull(b, field)) {
					// record
					this.recordInjectionInfo();
					this.parameters.setInjected(true);
					return true;

				}
			}
		}

		return false;
	}

	private boolean injectFieldWithNull(Body b, SootField field) {
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
						List<Stmt> stmts=this.getFieldStatementsByNull(b,field);
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
	private List<Stmt> getFieldStatementsByNull(Body b,SootField field) {
		Stmt assignNullStmt=null;
		List<Stmt> stmts=new ArrayList<Stmt>();
	
		assignNullStmt=Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(),field.makeRef()), NullConstant.v());	
		stmts.add(assignNullStmt);
		return stmts;
	}


	private boolean injectLocalWithNull(Body b, Local local) {
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
						List<Stmt> stmts=this.getLocalStatementsByNull(local);
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

	private List<Stmt> getLocalStatementsByNull(Local local) {
		List<Stmt> stmts=new ArrayList<Stmt>();
		Stmt valueChangeStmt=null;
	
		valueChangeStmt=Jimple.v().newAssignStmt(local, NullConstant.v());	
		stmts.add(valueChangeStmt);
		return stmts;
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
		System.out.println("HHHH");
		this.recorder.info(sBuffer.toString());
	}

	private Map<String, List<Object>> getAllVariables(Body b) {
		Map<String, List<Object>> result = new HashMap<String, List<Object>>();
		List<Local> tmpPLocals = b.getParameterLocals();
		List<Object> pLocals = new ArrayList<Object>();
		Chain<Local> tmpLocals = b.getLocals();
		List<Object> locals = new ArrayList<Object>();
		Chain<SootField> tmpFields = b.getMethod().getDeclaringClass().getFields();
		List<Object> fields = new ArrayList<Object>();

		int i;
		for (i = 0; i < tmpPLocals.size(); i++) {
			Local local = tmpPLocals.get(i);
			if (this.isTargetedType(local.getType().toString()) && (!local.getName().startsWith("$"))) {
				pLocals.add(local);
			}
		}

		Iterator<Local> tmpLocalsItr = tmpLocals.iterator();
		while (tmpLocalsItr.hasNext()) {
			Local local = tmpLocalsItr.next();
			if (!pLocals.contains(local)) {
				if (this.isTargetedType(local.getType().toString()) && (!local.getName().startsWith("$"))) {
					locals.add(local);
				}
			}

		}

		// actually soot cannot get Class Fields
		Iterator<SootField> tmpFieldItr = tmpFields.iterator();
		while (tmpFieldItr.hasNext()) {
			SootField field = tmpFieldItr.next();
			logger.info("Field name:　" + field.getName() + " Field type: " + field.getType().toString());
			if (this.isTargetedType(field.getType().toString()) && (!field.getName().startsWith("$"))) {
				fields.add(field);
			}
		}
		result.put("parameter", pLocals);
		result.put("local", locals);
		result.put("field", fields);
		return result;
	}

	private boolean isTargetedType(String typeName) {
		// only non-primitive types are included
		String[] targetTypes = { "boolean", "byte", "short", "int", "long", "float", "double", "java.lang.String" };
		List<String> list = Arrays.asList(targetTypes);
		if (list.contains(typeName)) {
			return false;
		} else {
			return true;
		}
	}

	// we should decide whether choose return as a scope
	private List<String> getTargetScope(String variableScope) {
		List<String> scopes = new ArrayList<String>();
		if ((variableScope == null) || (variableScope == "")) {
			scopes.add("local");
			scopes.add("field");
			scopes.add("parameter");
			scopes.add("return");
		} else {
			scopes.add(variableScope);
		}
		return scopes;
	}

}
