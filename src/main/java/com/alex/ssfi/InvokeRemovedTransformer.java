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
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.util.Chain;

public class InvokeRemovedTransformer extends BodyTransformer {
	private Map<String, String> injectInfo = new HashMap<String, String>();
	private RunningParameter parameters;
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);
	private final Logger recorder = LogManager.getLogger("inject_recorder");
	
	public InvokeRemovedTransformer(RunningParameter parameters) {
		this.parameters=parameters;
		
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
		this.injectInfo.put("FaultType", "EXCEPTION_UNCAUGHT_FAULT");
		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());
		List<String> actions = getTargetActions(this.parameters.getAction());

		// choose one scope to inject faults
		// if not specified, then randomly choose
		while (true) {
			int actionsSize = actions.size();
			if (actionsSize == 0) {
				logger.debug("Cannot find qualified scopes");
				break;
			}
			int actionIndex = new Random().nextInt(actionsSize);
			String action = actions.get(actionIndex);
			actions.remove(actionIndex);

			if (this.inject(b, action)) {
				break;
			}
		}

	}
	
	private boolean inject(Body b, String action) {
		//actually there is only one type of action
		//the action parameter is for future use
		try {
			Chain<Unit> units = b.getUnits();
			Iterator<Unit> unitItr=units.iterator();
			while(unitItr.hasNext()) {
				Stmt stmt=(Stmt)unitItr.next();
				//Currently we directly choose the first invokeStmt to delete
				if(stmt instanceof InvokeStmt) {
					InvokeStmt invStmt=(InvokeStmt)stmt;
					
					List<Stmt> actStmts = this.createActivateStatement(b);
					for (int i = 0; i < actStmts.size(); i++) {
						if (i == 0) {
							units.insertBefore(actStmts.get(i), invStmt);
						} else {
							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
						}
					}
					units.remove(invStmt);
					this.recordInjectionInfo();
					this.parameters.setInjected(true);
					return true;
				}
			}
			
				return false;

		} catch (Exception e) {
			logger.error(e.getMessage());
			return false;
		}
	}
	
	private List<String> getTargetActions(String action) {
		// TODO Auto-generated method stub
		List<String> actions = new ArrayList<String>();
		if ((action == null) || (action == "")) {
			actions.add("remove");
		} else {
			actions.add(action);
		}
		return actions;
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


}
