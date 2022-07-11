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
    }

    // 记录注入信息
    protected void recordInjectionInfo() {
        this.injectInfo.put("ComponentName", this.parameters.getComponentName()); // 记录目标组件名
        this.injectInfo.put("JarName", this.parameters.getJarName()); // 记录目标包名
        StringBuffer sBuffer = new StringBuffer();
        sBuffer.append("ID:");
        sBuffer.append(this.parameters.getID()); // 记录faultID
        for (Map.Entry<String, String> entry : this.injectInfo.entrySet()) { // 记录故障注入信息的每一项
            sBuffer.append("\t");
            sBuffer.append(entry.getKey());
            sBuffer.append(":");
            sBuffer.append(entry.getValue());
        }
        this.recorder.info(sBuffer.toString());
    }

    // 格式化注入信息
    protected String formatInjectionInfo() {
        StringBuffer sBuffer = new StringBuffer();
        sBuffer.append("ID:");
        sBuffer.append(this.parameters.getID());  // 记录faultID
        for (Map.Entry<String, String> entry : this.injectInfo.entrySet()) { // 记录故障注入信息的每一项
            sBuffer.append("\t");
            sBuffer.append(entry.getKey());
            sBuffer.append(":");
            sBuffer.append(entry.getValue());
        }
        return sBuffer.toString(); // 返回注入信息字符串
    }

    /*
     * Print this activation information to a file
     */
    // 构建激活语句
    protected List<Stmt> createActivateStatement(Body b) {
        SootClass fWriterClass = Scene.v().getSootClass("java.io.FileWriter"); // 获取"java.io.FileWriter"类
        SootClass stringClass = Scene.v().getSootClass("java.lang.String");// 获取"java.lang.String"类
        Local writer = Jimple.v().newLocal("actWriter", RefType.v(fWriterClass)); // 创建局部对象 FileWriter actWriter
        b.getLocals().add(writer); // 将"actWriter"添加到b的域
        // create a local variable to store the value time
        Local mstime = Jimple.v().newLocal("mstime", LongType.v());  // 创建局部对象 long mstime
        Local mstimeS = Jimple.v().newLocal("mstimeS", RefType.v(stringClass)); // 创建局部对象 String mstimeS
        b.getLocals().add(mstime); // 将"mstime"添加到b的域
        b.getLocals().add(mstimeS); // 将"mstimeS"添加到b的域
        SootMethod constructor = fWriterClass.getMethod("void <init>(java.lang.String,boolean)"); // 获取"FileWriter"的构造函数
        SootMethod stringConstructor = stringClass.getMethod("void <init>()"); // 获取"String"的构造函数
        SootMethod printMethod = Scene.v().getMethod("<java.io.Writer: void write(java.lang.String)>"); // 获取打印函数"java.io.Writer: void write(java.lang.String)"
        SootMethod closeMethod = Scene.v().getMethod("<java.io.OutputStreamWriter: void close()>"); // 获取关闭函数"java.io.OutputStreamWriter: void close()"
        SootMethod currentTimeMethod = Scene.v().getMethod("<java.lang.System: long currentTimeMillis()>"); // 获取当前时间函数"java.lang.System: long currentTimeMillis()"
        SootMethod convertLong2StringMethod = Scene.v().getMethod("<java.lang.Long: java.lang.String toString(long)>"); // 获取long2String函数"java.lang.Long: java.lang.String toString(long)"

        AssignStmt newStmt = Jimple.v().newAssignStmt(writer, Jimple.v().newNewExpr(RefType.v("java.io.FileWriter")));//赋值语句 actWriter = new FileWriter
        AssignStmt newStringInitStmt = Jimple.v().newAssignStmt(mstimeS, Jimple.v().newNewExpr(RefType.v(stringClass)));//赋值语句 mstimeS = new String
        // InvokeStmt invStmt =
        // Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer,
        // constructor.makeRef(),
        // StringConstant.v(this.parameters.getOutput() + File.separator +
        // "logs/activation.log"), IntConstant.v(1)));
        InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(writer, constructor.makeRef(),
                StringConstant.v(this.parameters.getActivationLogFile()), IntConstant.v(1))); // 调用语句 actWriter = FileWriter(ActivationLogFile,1)
        InvokeStmt invStringInitStmt = Jimple.v()
                .newInvokeStmt(Jimple.v().newSpecialInvokeExpr(mstimeS, stringConstructor.makeRef()));// 调用语句 mstimeS = String()

        // generate time
        AssignStmt timeStmt = Jimple.v().newAssignStmt(mstime,
                Jimple.v().newStaticInvokeExpr(currentTimeMethod.makeRef())); // 赋值语句 mstime = currentTimeMillis()

        // convert long time to string time
        AssignStmt long2String = Jimple.v().newAssignStmt(mstimeS,
                Jimple.v().newStaticInvokeExpr(convertLong2StringMethod.makeRef(), mstime)); // 赋值语句 mstimeS = mstime.toString()
        // print time
        InvokeStmt logTimeStmt = Jimple.v()
                .newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(), mstimeS)); // 调用语句 actWriter.write(mstimeS)
        // print injection ID
        InvokeStmt logStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, printMethod.makeRef(),
                StringConstant.v(":" + this.parameters.getID() + "\n")));// 调用语句 actWriter.write(":"+ID+"\n")
        InvokeStmt closeStmt = Jimple.v().newInvokeStmt(Jimple.v().newVirtualInvokeExpr(writer, closeMethod.makeRef())); // 调用语句 actWriter.close()

        List<Stmt> statements = new ArrayList<Stmt>();
        statements.add(newStmt); //记录赋值语句 actWriter = new FileWriter
        statements.add(newStringInitStmt);//记录赋值语句 mstimeS = new String
        statements.add(invStmt); // 记录调用语句 actWriter = FileWriter(ActivationLogFile,1)
        statements.add(invStringInitStmt);// 记录调用语句 mstimeS = String()
        statements.add(timeStmt);// 记录赋值语句 mstime = currentTimeMillis()
        statements.add(long2String);// 记录赋值语句 mstimeS = mstime.toString()
        statements.add(logTimeStmt); // 记录调用语句 actWriter.write(mstimeS)
        statements.add(logStmt);// 记录调用语句 actWriter.write(":"+faultID+"\n")
        statements.add(closeStmt);// 记录调用语句 actWriter.close()

        if (this.parameters.getActivationMode().equals("CUSTOMIZED")) { //若激活模式为"CUSTOMIZED"
            // call user-defined method

            SootClass activationHelperClass = Scene.v().getSootClass(this.parameters.getCustomizedActivation()); // 从配置文件customizedActivation项，获取激活帮助类
            SootMethod activationMarker = activationHelperClass.getMethodByNameUnsafe("activate"); // 获取"activate"方法，作为激活故障时的操作
            // tmpActivated
            // convert long time to string time
            InvokeStmt markActivationStmt =
                    Jimple.v().newInvokeStmt(Jimple.v().newStaticInvokeExpr(
                            activationMarker.makeRef(), // 方法名
                            StringConstant.v(this.parameters.getID()))); // 调用语句 CustomActivationController.activate(faultID)
            statements.add(markActivationStmt); // 记录调用语句 CustomActivationController.activate(faultID)
            return statements; //返回激活语句

        }
        if (!this.parameters.getActivationMode().equals("ALWAYS")) { // 若激活模式为"ALWAYS"
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());  // 在目标代码内获取"sootActivated"字段
            AssignStmt injectedStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(activated.makeRef()),
                    IntConstant.v(1)); // 赋值语句 sootActivated = 1
            statements.add(injectedStmt); //返回激活语句
        }

        return statements; //返回激活语句
    }

    // 获取前置检查语句，根据激活模式和参数，设置激活flag和目标执行次数
    protected List<Stmt> getPrecheckingStmts(Body b) {
        this.injectInfo.put("ActivationMode", "" + this.parameters.getActivationMode());

        List<Stmt> preCheckStmts = new ArrayList<Stmt>();
        String activationMode = this.parameters.getActivationMode(); // 获取激活模式
        SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v()); // 获取soot执行计数器

        if (exeCounter == null) { // 若目标Body未声明"sootCounter"字段
            exeCounter = new SootField("sootCounter", LongType.v(), Modifier.STATIC); // 在目标Body内声明"sootCounter"字段
            b.getMethod().getDeclaringClass().addField(exeCounter); // 并将该soot执行计数器添加到目标Body所在类的域内
        }
        Local tmp = Jimple.v().newLocal("sootTmpExeCounter", LongType.v()); // 获取soot局部执行计数器
        b.getLocals().add(tmp); // 并在该soot局部执行计数器添加到目标Body的域内
        AssignStmt copyStmt = Jimple.v().newAssignStmt(tmp, Jimple.v().newStaticFieldRef(exeCounter.makeRef())); // 赋值语句 sootTmpExeCounter = sootCounter
        AddExpr addExpr = Jimple.v().newAddExpr(tmp, LongConstant.v(1)); // 加法语句 sootTmpExeCounter + 1
        AssignStmt addStmt = Jimple.v().newAssignStmt(tmp, addExpr); // 赋值语句 sootTmpExeCounter = sootTmpExeCounter + 1
        AssignStmt assignStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(exeCounter.makeRef()), tmp); // 赋值语句 sootCounter = sootTmpExeCounter

        preCheckStmts.add(copyStmt); // 记录复制语句 sootTmpExeCounter = sootCounter
        preCheckStmts.add(addStmt); // 记录加法语句 sootTmpExeCounter = sootTmpExeCounter + 1
        preCheckStmts.add(assignStmt); // 记录赋值语句 sootCounter = sootTmpExeCounter

        if (activationMode.equals("ALWAYS")) { // 若激活模式为"ALWAYS"
            // only add counter
            // done by above
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v()); // 在目标代码内获取"sootActivated"字段
            if (activated == null) { // 若目标代码内不存在"sootActivated"字段
                activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC); // 创建"sootActivated"静态字段
                b.getMethod().getDeclaringClass().addField(activated); // 在目标代码内声明"sootActivated"字段
            }
        } else if (activationMode.equals("FIRST")) {// 若激活模式为"FIRST"
            // add flag
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());// 在目标代码内获取"sootActivated"字段
            if (activated == null) { // 若目标代码内不存在"sootActivated"字段
                activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);// 创建"sootActivated"静态字段
                b.getMethod().getDeclaringClass().addField(activated);// 在目标代码内声明"sootActivated"字段
            }
        } else if (activationMode.equals("RANDOM")) {// 若激活模式为"RANDOM"
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());// 在目标代码内获取"sootActivated"字段
            if (activated == null) {// 若目标代码内不存在"sootActivated"字段
                activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);// 创建"sootActivated"静态字段
                b.getMethod().getDeclaringClass().addField(activated);// 在目标代码内声明"sootActivated"字段
                int targetCounter = new Random(System.currentTimeMillis()).nextInt(this.parameters.getActivationRate()) // 在ActivationRate范围内，随机选择下一次激活故障的执行次数
                        + 1;
                this.injectInfo.put("ExeIndex", "" + targetCounter);
                exeCounter.addTag(new RandomTag(targetCounter)); // 记录目标执行次数
            }
        } else if (activationMode.equals("FIXED")) {// 若激活模式为"FIXED"
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());// 在目标代码内获取"sootActivated"字段
            if (activated == null) {// 若目标代码内不存在"sootActivated"字段
                activated = new SootField("sootActivated", BooleanType.v(), Modifier.STATIC);// 创建"sootActivated"静态字段
                b.getMethod().getDeclaringClass().addField(activated);// 在目标代码内声明"sootActivated"字段
                int targetCounter = parameters.getActivationIndex(); // 获取指定目标执行次数，在该次执行时激活故障
                this.injectInfo.put("ActivationIndex", "" + targetCounter);
                exeCounter.addTag(new RandomTag(targetCounter)); // 记录目标执行次数
            }
        } else if (activationMode.equals("CUSTOMIZED")) {
            // in this activation mode, the whole logic is controlled by user-defined code
            // SSFI doesn't do anything
        }
        return preCheckStmts; // 返回前置检查语句，
    }

    // 获取条件语句，根据激活flag、目标执行次数、当前执行次数设置故障激活的条件语句
    protected List<Stmt> getConditionStmt(Body b, Unit failTarget) {
        this.injectInfo.put("LineNumber", String.valueOf(failTarget.getJavaSourceStartLineNumber()));
        this.injectInfo.put("StringUnit", failTarget.toString());
        List<Stmt> conditionStmts = new ArrayList<Stmt>();
        String activationMode = this.parameters.getActivationMode(); // 获取激活模式
        Local tmpActivated = Jimple.v().newLocal("sootActivated", BooleanType.v()); // 创建局部布尔变量"sootActivated"
        b.getLocals().add(tmpActivated); // 将局部变量"sootActivated"，添加到b的域

        if (activationMode.equals("ALWAYS")) { // 若激活模式为"ALWAYS"
            // if false, go to the default target
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v()); // 获取字段布尔变量"sootActivated"
            AssignStmt copyStmt = Jimple.v().newAssignStmt(tmpActivated,
                    Jimple.v().newStaticFieldRef(activated.makeRef())); // 复制语句 sootActivated = this.sootActivated
            IfStmt ifStmt = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget);  // 构建if语句 if (sootActivated) {failTarget}

            conditionStmts.add(copyStmt); // 记录复制语句 sootActivated = this.sootActivated
            conditionStmts.add(ifStmt); // 记录if语句 if (sootActivated) {failTarget}

        } else if (activationMode.equals("FIRST")) { // 若激活模式为"ALWAYS"
            // add flag
            Local tmpCounterRef = Jimple.v().newLocal("sootCounterRef", LongType.v()); // 创建局部变量"sootCounterRef"
            b.getLocals().add(tmpCounterRef); // 将局部变量"sootCounterRef"，添加到b的域
            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v());// 获取字段布尔变量"sootActivated"
            AssignStmt copyStmt1 = Jimple.v().newAssignStmt(tmpActivated,
                    Jimple.v().newStaticFieldRef(activated.makeRef()));// 复制语句 sootActivated = this.sootActivated

            SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v()); // 获取字段变量"sootCounter"
            AssignStmt copyStmt2 = Jimple.v().newAssignStmt(tmpCounterRef,
                    Jimple.v().newStaticFieldRef(exeCounter.makeRef())); // 复制语句 sootCounterRef = this.sootCounter
            IfStmt ifStmt2 = Jimple.v().newIfStmt(Jimple.v().newNeExpr(tmpCounterRef, LongConstant.v(1)), failTarget); // if语句 if(sootCounterRef!=1){failTarget}
            // if already activated, directly go to the default target unit
            IfStmt ifStmt1 = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget); // if语句 if(sootActivated){failTarget}
            conditionStmts.add(copyStmt1);// 记录复制语句 sootActivated = this.sootActivated
            conditionStmts.add(copyStmt2);// 记录复制语句 sootCounterRef = this.sootCounter
            conditionStmts.add(ifStmt1); // 记录if语句 if(sootActivated){failTarget}
            conditionStmts.add(ifStmt2); // 记录if语句 if(sootCounterRef!=1){failTarget}

        } else if (activationMode.equals("RANDOM") || activationMode.equals("FIXED")) { // 若激活模式为"RANDOM"或"FIXED"
            Local tmpCounterRef = Jimple.v().newLocal("sootCounterRef", LongType.v()); // 创建局部变量"sootCounterRef"
            b.getLocals().add(tmpCounterRef);// 将局部变量"sootCounterRef"，添加到b的域

            SootField activated = b.getMethod().getDeclaringClass().getFieldUnsafe("sootActivated", BooleanType.v()); // 获取字段变量"sootActivated"
            AssignStmt copyStmt1 = Jimple.v().newAssignStmt(tmpActivated,
                    Jimple.v().newStaticFieldRef(activated.makeRef())); // 复制语句 sootActivated = this.sootActivated
            SootField exeCounter = b.getMethod().getDeclaringClass().getFieldUnsafe("sootCounter", LongType.v()); // 获取字段变量"sootCounter"
            AssignStmt copyStmt2 = Jimple.v().newAssignStmt(tmpCounterRef,
                    Jimple.v().newStaticFieldRef(exeCounter.makeRef())); // 复制语句 sootCounterRef = this.sootCounter

            RandomTag randomTag = (RandomTag) exeCounter.getTag("sootTargetCounter"); // 获取目标执行次数对象
            long targetCounter = randomTag.getTargetCounter(); // 获取目标执行次数
            IfStmt ifStmt2 = Jimple.v().newIfStmt(Jimple.v().newNeExpr(tmpCounterRef, LongConstant.v(targetCounter)),
                    failTarget); // if语句 if(sootCounterRef!=sootTargetCounter) {failTarget}
            IfStmt ifStmt1 = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget); // if语句 if(sootActivated){failTarget}

            conditionStmts.add(copyStmt1); // 记录复制语句 sootActivated = this.sootActivated
            conditionStmts.add(copyStmt2);// 记录复制语句 sootCounterRef = this.sootCounter
            conditionStmts.add(ifStmt1); // 记录if语句 if(sootActivated){failTarget}
            conditionStmts.add(ifStmt2);// 记录if语句 if(sootCounterRef!=sootTargetCounter) {failTarget}

        } else if (activationMode.equals("CUSTOMIZED")) { // 若激活模式为"CUSTOMIZED"
            // user-defined activation checking logic, SSFI just calls that user-defined
            // method
            SootClass activationHelperClass = Scene.v().getSootClass(this.parameters.getCustomizedActivation());// 导入配置文件中customizedActivation对应的类
            SootMethod activationChecker = activationHelperClass.getMethodByNameUnsafe("shouldActivate"); // 导入shouldActivate方法，作为激活故障的条件判断方法
            AssignStmt checkActivationResult = Jimple.v().newAssignStmt(tmpActivated,
                    Jimple.v().newStaticInvokeExpr( // 函数调用
                            activationChecker.makeRef(), // 方法名
                            StringConstant.v(this.parameters.getID()),// 参数 faultID
                            StringConstant.v(failTarget.toString()), // 参数 unit
                            IntConstant.v(failTarget.getJavaSourceStartLineNumber()) // 参数 lineNumber
                    )); // 赋值语句 sootActivated = CustomActivationController.shouldActivate(faultID,unit,lineNumber)
            IfStmt ifStmt1 = Jimple.v().newIfStmt(Jimple.v().newEqExpr(tmpActivated, IntConstant.v(1)), failTarget); // 条件语句 if(sootActivated){failTarget}
            conditionStmts.add(checkActivationResult); // 记录赋值语句 sootActivated = CustomActivationController.shouldActivate(faultID,unit,lineNumber)
            conditionStmts.add(ifStmt1); // 条件语句 if(sootActivated==0){failTarget}
        }
        return conditionStmts; // 返回条件语句
    }
}