package com.alex.ssfi;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

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
		this.injectInfo.put("ComponentName", this.parameters.getComponentName());
		this.injectInfo.put("JarName", this.parameters.getJarName());
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
//		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
//				StringConstant.v(this.parameters.getOutput() + File.separator + "logs/activation.log"), IntConstant.v(1)));
		InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
				StringConstant.v(this.parameters.getActivationLogFile()), IntConstant.v(1)));
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
		if (!this.parameters.getActivationMode().equals("always")) {
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			AssignStmt injectedStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(activated.makeRef()),
					IntConstant.v(1));
			statements.add(injectedStmt);
		}

		return statements;
	}

	protected List<Stmt> getPrecheckingStmts(Body b) {
		this.injectInfo.put("ActivationMode", "" + this.parameters.getActivationMode());

		List<Stmt> preCheckStmts = new ArrayList<Stmt>();
		String activationMode = this.parameters.getActivationMode();
		SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());

		if (exeCounter == null) {
			exeCounter = new SootField("sootCounter", LongType.v(), Modifier.STATIC);
			b.getMethod().getDeclaringClass().addField(exeCounter);
		}
		Local tmp = Jimple.v().newLocal("sootTmpExeCounter", LongType.v());
		b.getLocals().add(tmp);
		AssignStmt copyStmt = Jimple.v().newAssignStmt(tmp, Jimple.v().newStaticFieldRef(exeCounter.makeRef()));
		AddExpr addExpr = Jimple.v().newAddExpr(tmp, LongConstant.v(1));
		AssignStmt addStmt = Jimple.v().newAssignStmt(tmp, addExpr);
		AssignStmt assignStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(exeCounter.makeRef()), tmp);

		// for debug
//		Local tmpRef = Jimple.v().newLocal("tmpRef", RefType.v("java.io.PrintStream"));
//		b.getLocals().add(tmpRef);
//		Stmt debugCopy = Jimple.v().newAssignStmt(tmpRef, Jimple.v()
//				.newStaticFieldRef(Scene.v().getField("<java.lang.System: java.io.PrintStream out>").makeRef()));
//		SootMethod toCall = Scene.v().getMethod("<java.io.PrintStream: void println(long)>");
//		Stmt printStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(tmpRef, toCall.makeRef(), tmp));

		preCheckStmts.add(copyStmt);
//		preCheckStmts.add(debugCopy);
//		preCheckStmts.add(printStmt);
		preCheckStmts.add(addStmt);
		preCheckStmts.add(assignStmt);

		if (activationMode.equals("always")) {
			// only add counter
			// done by above
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			if (activated == null) {
				activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);
				b.getMethod().getDeclaringClass().addField(activated);
			}

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
				int targetCounter = new Random(System.currentTimeMillis()).nextInt(this.parameters.getActivationRate())
						+ 1;
				this.injectInfo.put("ExeIndex", "" + targetCounter);
				exeCounter.addTag(new RandomTag(targetCounter));
			}

		}

		return preCheckStmts;

	}

	protected List<Stmt> getConditionStmt(Body b, Unit failTarget) {
		List<Stmt> conditionStmts = new ArrayList<Stmt>();
		String activationMode = this.parameters.getActivationMode();
		Local tmpActivated = Jimple.v().newLocal("sootActivated", BooleanType.v());
		b.getLocals().add(tmpActivated);

		if (activationMode.equals("always")) {
			// if false, go to the default target
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			AssignStmt copyStmt = Jimple.v().newAssignStmt(tmpActivated,
					Jimple.v().newStaticFieldRef(activated.makeRef()));

			IfStmt ifStmt = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget);
			conditionStmts.add(copyStmt);
			conditionStmts.add(ifStmt);

		} else if (activationMode.equals("first")) {
			// add flag
			Local tmpCounterRef = Jimple.v().newLocal("sootCounterRef", LongType.v());
			b.getLocals().add(tmpCounterRef);
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			AssignStmt copyStmt1 = Jimple.v().newAssignStmt(tmpActivated,
					Jimple.v().newStaticFieldRef(activated.makeRef()));

			SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());
			AssignStmt copyStmt2 = Jimple.v().newAssignStmt(tmpCounterRef,
					Jimple.v().newStaticFieldRef(exeCounter.makeRef()));
			IfStmt ifStmt2 = Jimple.v().newIfStmt(Jimple.v().newNeExpr(tmpCounterRef, LongConstant.v(1)), failTarget);
			// if already activated, directly go to the default target unit
			IfStmt ifStmt1 = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget);
			conditionStmts.add(copyStmt1);
			conditionStmts.add(copyStmt2);
			conditionStmts.add(ifStmt1);
			conditionStmts.add(ifStmt2);

		} else if (activationMode.equals("random")) {
			Local tmpCounterRef = Jimple.v().newLocal("sootCounterRef", LongType.v());
			b.getLocals().add(tmpCounterRef);
			SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());
			AssignStmt copyStmt1 = Jimple.v().newAssignStmt(tmpActivated,
					Jimple.v().newStaticFieldRef(activated.makeRef()));
			SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v());
			AssignStmt copyStmt2 = Jimple.v().newAssignStmt(tmpCounterRef,
					Jimple.v().newStaticFieldRef(exeCounter.makeRef()));

			RandomTag randomTag = (RandomTag) exeCounter.getTag("sootTargetCounter");
			long targetCounter = randomTag.getTargetCounter();
			IfStmt ifStmt2 = Jimple.v().newIfStmt(Jimple.v().newNeExpr(tmpCounterRef, LongConstant.v(targetCounter)),
					failTarget);
			IfStmt ifStmt1 = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget);

			conditionStmts.add(copyStmt1);

			conditionStmts.add(copyStmt2);
			conditionStmts.add(ifStmt1);
			conditionStmts.add(ifStmt2);

		}
		return conditionStmts;
	}

}
