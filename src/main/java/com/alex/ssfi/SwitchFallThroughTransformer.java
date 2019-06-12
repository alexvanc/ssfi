package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

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
import soot.UnitBox;
import soot.jimple.AssignStmt;
import soot.jimple.Expr;
import soot.jimple.GeExpr;
import soot.jimple.GtExpr;
import soot.jimple.IfStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.LeExpr;
import soot.jimple.LtExpr;
import soot.jimple.NopStmt;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.SwitchStmt;
import soot.util.Chain;

public class SwitchFallThroughTransformer extends BodyTransformer {
	private Map<String, String> injectInfo = new HashMap<String, String>();
	private RunningParameter parameters;
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);
	private final Logger recorder = LogManager.getLogger("inject_recorder");
	
	public SwitchFallThroughTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "SWITCH_FALLTHROUGH_FAULT");
		this.injectInfo.put("Package", b.getMethod().getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", b.getMethod().getDeclaringClass().getName());

		Chain<Unit> units=b.getUnits();
		Iterator<Unit> unitItr=units.iterator();
		// choose one scope to inject faults
		// if not specified, then randomly choose
		while (unitItr.hasNext()) {
			Stmt stmt=(Stmt)unitItr.next();
			if(stmt instanceof SwitchStmt) {
				if(this.inject(b,stmt)) {
					List<Stmt> actStmts = this.createActivateStatement(b);
					for (int i = 0; i < actStmts.size(); i++) {
						if (i == 0) {
							units.insertBefore(actStmts.get(i), stmt);
						} else {
							units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
						}
					}
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
					break;
				}
			}
		}

	}
	
	private boolean inject(Body b, Stmt stmt) {
		// TODO Auto-generated method stub
		try {
		SwitchStmt switchStmt=(SwitchStmt)stmt;
		List<Unit> targets=switchStmt.getTargets();
		for(int i=0;i<targets.size();i++) {
			Unit currentCase=targets.get(i);
			Unit nextCase=targets.get(i+1);
			List<UnitBox> unitBoxes=nextCase.getBoxesPointingToThis();
			boolean contained=false;
			for(int j=0;j<unitBoxes.size();j++) {
				Unit tmp=unitBoxes.get(i).getUnit();
				if(tmp.equals(currentCase)) {
					contained=true;
				}
			}
			if(contained) {//it's fallthrough, we need to add a break stmt
				Chain<Unit> units=b.getUnits();
				Iterator<Unit> unitItr=units.snapshotIterator();
				while(unitItr.hasNext()) {
					Unit tmp=unitItr.next();
					if(tmp.equals(currentCase)) {
						while(unitItr.hasNext()) {
							Stmt tmpStmt=(Stmt) unitItr.next();
							if(tmpStmt.getUnitBoxes().size()!=0) {//the last stmt of this case
								NopStmt noStmt=Jimple.v().newNopStmt();
								units.insertAfter(noStmt, tmpStmt);
								return true;
							}
						}
					}
				}
				
			}else {//the two cases are separated, we need to delete the break stmt of the currentCase
				Chain<Unit> units=b.getUnits();
				Iterator<Unit> unitItr=units.snapshotIterator();
				while(unitItr.hasNext()) {
					Unit tmp=unitItr.next();
					if(tmp.equals(currentCase)) {
						while(unitItr.hasNext()) {
							Stmt tmpStmt=(Stmt) unitItr.next();
							if(tmpStmt instanceof NopStmt) {
								units.remove(tmpStmt);
								return true;
							}
							
						}
					}
				}
			}
			
		}

		}catch (Exception e) {
			logger.error(e.getMessage());
		}
		return false;
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
