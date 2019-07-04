package com.alex.ssfi;

import java.io.File;
import java.rmi.activation.Activatable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.security.auth.login.FailedLoginException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.BodyTransformer;
import soot.BooleanType;
import soot.Local;
import soot.LongType;
import soot.Modifier;
import soot.RefType;
import soot.Scene;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;
import soot.jimple.AddExpr;
import soot.jimple.AssignStmt;
import soot.jimple.IfStmt;
import soot.jimple.IntConstant;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.LongConstant;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;

public abstract class BasicTransformer extends BodyTransformer {
	protected Map<String, String> injectInfo = new HashMap<String, String>();
	protected RunningParameter parameters;
	protected final Logger recorder = LogManager.getLogger("inject_recorder");
	protected List<SootMethod> allQualifiedMethods = null;

	public BasicTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	public BasicTransformer() {
	}

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
		Local writer = Jimple.v().newLocal("actWriter", RefType.v(fWriterClass));
		b.getLocals().add(writer);
		SootMethod constructor = fWriterClass.getMethod("void <init>(java.lang.String,boolean)");
		SootMethod printMethod = Scene.v().getMethod("<java.io.Writer: void write(java.lang.String)>");
		SootMethod closeMethod = Scene.v().getMethod("<java.io.OutputStreamWriter: void close()>");

		AssignStmt newStmt = Jimple.v().newAssignStmt(writer, Jimple.v().newNewExpr(RefType.v("java.io.FileWriter")));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
				StringConstant.v(this.parameters.getOutput() + File.separator + "activation.txt"), IntConstant.v(1)));
		InvokeStmt logStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(),
				StringConstant.v(this.parameters.getID() + "\n")));
		InvokeStmt closeStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, closeMethod.makeRef()));

		List<Stmt> statements = new ArrayList<Stmt>();
		statements.add(newStmt);
		statements.add(invStmt);
		statements.add(logStmt);
		statements.add(closeStmt);
		return statements;
	}

	protected List<Stmt> getPrecheckingStmts(Body b) {

		List<Stmt> preCheckStmts = new ArrayList<Stmt>();
		String activationMode = this.parameters.getActivationMode();
		SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());

		if (exeCounter == null) {
			exeCounter = new SootField("sootCounter", LongType.v(), Modifier.STATIC);
			b.getMethod().getDeclaringClass().addField(exeCounter);
		}
		Local tmp = Jimple.v().newLocal("sootTmp", LongType.v());
		b.getLocals().add(tmp);
		AssignStmt copyStmt = Jimple.v().newAssignStmt(tmp, Jimple.v().newStaticFieldRef(exeCounter.makeRef()));
		AddExpr addExpr = Jimple.v().newAddExpr(tmp, LongConstant.v(1));
		AssignStmt addStmt = Jimple.v().newAssignStmt(tmp, addExpr);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(exeCounter.makeRef()), tmp);
		preCheckStmts.add(copyStmt);
		preCheckStmts.add(addStmt);
		preCheckStmts.add(assignStmt);

		if (activationMode.equals("always")) {
			// only add counter
			// done by above

		} else if (activationMode.equals("first")) {
			// add flag
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			if (activated == null) {
				activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);
				b.getMethod().getDeclaringClass().addField(activated);
			}

		} else if (activationMode.equals("random")) {
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			if (activated == null) {
				activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);
				b.getMethod().getDeclaringClass().addField(activated);
				int targetCounter = new Random(System.currentTimeMillis()).nextInt(this.parameters.getActivationRate());
				exeCounter.addTag(new RandomTag(targetCounter));
			}

		}

		return preCheckStmts;

	}

	protected List<Stmt> getConditionStmt(Body b, Unit failTarget) {
		List<Stmt> conditionStmts=new ArrayList<Stmt>();
		String activationMode = this.parameters.getActivationMode();
//		IfStmt ifStmt=null;

		if (activationMode.equals("always")) {
			// if false, go to the default target
			IfStmt ifStmt = Jimple.v().newIfStmt(IntConstant.v(0), failTarget);
			conditionStmts.add(ifStmt);

		} else if (activationMode.equals("first")) {
			// add flag
			SootField activated=b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());
			IfStmt ifStmt2 = Jimple.v().newIfStmt(
					Jimple.v().newNeExpr(Jimple.v().newStaticFieldRef(exeCounter.makeRef()), LongConstant.v(1)),
					failTarget);	
			//if already activated, directly go to the default target unit
			IfStmt ifStmt1=Jimple.v().newIfStmt(Jimple.v().newStaticFieldRef(activated.makeRef()), failTarget);
			conditionStmts.add(ifStmt1);
			conditionStmts.add(ifStmt2);

		} else if (activationMode.equals("random")) {
			SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());
			SootField activated=b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());

			RandomTag randomTag = (RandomTag) exeCounter.getTag("sootTargetCounter");
			long targetCounter = randomTag.getTargetCounter();
			IfStmt ifStmt2 = Jimple.v().newIfStmt(Jimple.v().newNeExpr(
					Jimple.v().newStaticFieldRef(exeCounter.makeRef()), LongConstant.v(targetCounter)), failTarget);
			IfStmt ifStmt1=Jimple.v().newIfStmt(Jimple.v().newStaticFieldRef(activated.makeRef()), failTarget);
			conditionStmts.add(ifStmt1);
			conditionStmts.add(ifStmt2);

		}
		return conditionStmts;
	}

}
