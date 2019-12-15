package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.BodyTransformer;
import soot.Local;
import soot.LongType;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.jimple.AssignStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;

public abstract class BasicTransformer extends BodyTransformer {
	protected Map<String, String> injectInfo = new HashMap<String, String>();
	protected RunningParameter parameters;
	protected final Logger recorder = LogManager.getLogger("inject_recorder");

	public BasicTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	public BasicTransformer() {
	}

	// because we process projects in a class level
	// the methodIndex represents the method position of soot's traversing
	protected int methodIndex = 0;

	protected boolean foundTargetMethod = false;

	protected String targetMethodSubSignature;

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub

	}

	protected void recordInjectionInfo() {
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

	protected String formatInjectionInfo() {
		StringBuffer sBuffer = new StringBuffer();
		sBuffer.append("ID:");
		sBuffer.append(this.parameters.getID());
		for (Map.Entry<String, String> entry : this.injectInfo.entrySet()) {
			sBuffer.append("\t");
			sBuffer.append(entry.getKey());
			sBuffer.append(":");
			sBuffer.append(entry.getValue());
		}
		return sBuffer.toString();
	}

	/*
	 * Print this activation information to a file
	 */
	protected List<Stmt> createActivateStatement(Body b) {
		SootClass fWriterClass = Scene.v().getSootClass("java.io.FileWriter");
		SootClass stringClass=Scene.v().getSootClass("java.lang.String");
		Local writer = Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
		b.getLocals().add(writer);
		// create a local variable to store the value time
		Local mstime = Jimple.v().newLocal("mstime", LongType.v());
		Local mstimeS = Jimple.v().newLocal("mstimeS", RefType.v(stringClass));
		b.getLocals().add(mstime);
		b.getLocals().add(mstimeS);
		SootMethod constructor = fWriterClass.getMethod("void <init>(java.lang.String,boolean)");
		SootMethod stringConstructor = stringClass.getMethod("void <init>()");
		SootMethod printMethod = Scene.v().getMethod("<java.io.Writer: void write(java.lang.String)>");
		SootMethod closeMethod = Scene.v().getMethod("<java.io.OutputStreamWriter: void close()>");
		SootMethod currentTimeMethod = Scene.v().getMethod("<java.lang.System: long currentTimeMillis()>");
		SootMethod convertLong2StringMethod=Scene.v().getMethod("<java.lang.Long: java.lang.String toString(long)>");

		AssignStmt newStmt = Jimple.v().newAssignStmt(writer, Jimple.v().newNewExpr(RefType.v("java.io.FileWriter")));
		AssignStmt newStringInitStmt = Jimple.v().newAssignStmt(mstimeS, Jimple.v().newNewExpr(RefType.v(stringClass)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
				StringConstant.v(this.parameters.getOutput() + File.separator + "activation.txt"), IntConstant.v(1)));
		InvokeStmt invStringInitStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(mstimeS, stringConstructor.makeRef()));

		// generate time
		AssignStmt timeStmt = Jimple.v().newAssignStmt(mstime,
				Jimple.v().newStaticInvokeExpr(currentTimeMethod.makeRef()));
		
		// convert long time to string time
		AssignStmt long2String = Jimple.v().newAssignStmt(mstimeS,
				Jimple.v().newStaticInvokeExpr(convertLong2StringMethod.makeRef(),mstime));
		// print time
		InvokeStmt logTimeStmt = Jimple.v()
				.newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(), mstimeS));
		//print injection ID
		InvokeStmt logStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(),
				StringConstant.v(":" + this.parameters.getID() + "\n")));
		InvokeStmt closeStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, closeMethod.makeRef()));

		List<Stmt> statements = new ArrayList<Stmt>();
		statements.add(newStmt);
		statements.add(newStringInitStmt);
		statements.add(invStmt);
		statements.add(invStringInitStmt);
		statements.add(timeStmt);
		statements.add(long2String);
		statements.add(logTimeStmt);
		statements.add(logStmt);
		statements.add(closeStmt);
		return statements;
	}

}
