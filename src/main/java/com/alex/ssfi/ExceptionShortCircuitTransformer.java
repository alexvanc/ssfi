package com.alex.ssfi;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.Local;
import soot.RefType;
import soot.SootClass;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.AssignStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class ExceptionShortCircuitTransformer extends BasicTransformer {
    private final Logger logger = LogManager.getLogger(ExceptionShortCircuitTransformer.class);

    public ExceptionShortCircuitTransformer(RunningParameter parameters) {
        this.parameters = parameters;
    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {// 核心的回调方法
        while (!this.parameters.isInjected()) {
            // in this way, all the FIs are performed in the first function of this class
            SootMethod targetMethod = this.generateTargetMethod(b);// 获取待注入的Method
            if (targetMethod == null) {// 若获取不到待注入的Method，则返回上级程序
                return;
            }
            this.startToInject(targetMethod.getActiveBody());// 开始故障注入。对待注入的Method对应的Body进行故障注入
        }
    }

    // 对于目标Body进行故障注入
    private void startToInject(Body b) {
        // no matter this inject fails or succeeds, this targetMethod is already used
        SootMethod targetMethod = b.getMethod();// 获取目标Method
        this.injectInfo.put("FaultType", "EXCEPTION_SHORTCIRCUIT_FAULT");// 标记故障类型为，"EXCEPTION_SHORTCIRCUIT_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名
        List<String> scopes = getTargetScope(this.parameters.getVariableScope());// 获取目标域集合

        logger.debug("Try to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // randomly combine scope
        while (true) {
            int scopesSize = scopes.size();// 获取待注入的域集合
            if (scopesSize == 0) {
                logger.debug("Cannot find qualified scopes");
                break;
            }
            int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize);// 随机选择待注入的域的索引
            String scope = scopes.get(scopeIndex);// 获取待注入的域
            scopes.remove(scopeIndex);// 在待注入的域集合中，移除将要进行尝试的域
            this.injectInfo.put("VariableScope", scope);// 记录待注入的域

            // finally perform injection
            if (this.inject(b, scope)) {// 进行注入
                this.parameters.setInjected(true);// 标记注入成功
                this.recordInjectionInfo();// 记录注入信息
                logger.debug("Succeed to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package")
                        + " " + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                return;
            } else {
                logger.debug("Failed injection:+ " + this.formatInjectionInfo());
            }

            logger.debug("Fail to inject EXCEPTION_SHORTCIRCUIT_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
        }

    }

    // 获取待注入的Method
    private synchronized SootMethod generateTargetMethod(Body b) {
        if (this.allQualifiedMethods == null) {// 若符合要求的Method集合为null
            this.initAllQualifiedMethods(b);// 初始化符合要求的Method集合
        }
        int leftQualifiedMethodsSize = this.allQualifiedMethods.size();// 获取符合要求的Method集合的大小
        if (leftQualifiedMethodsSize == 0) {// 若符合要求的Method集合的大小为0，则返回null
            return null;
        }
        int randomMethodIndex = new Random(System.currentTimeMillis()).nextInt(leftQualifiedMethodsSize);// 随机选择待注入Method的索引
        SootMethod targetMethod = this.allQualifiedMethods.get(randomMethodIndex);// 获取待注入的Method
        this.allQualifiedMethods.remove(randomMethodIndex);// 在待注入Method的集合中，移除将要进行尝试的Method
        return targetMethod;// 返回待注入的Method
    }

    // for this fault type,we extract the methods with declared exceptions or with
    // try-catch blocks
    private void initAllQualifiedMethods(Body b) {
        List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();// 获取当前body能够获取的全部Method
        List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();// 声明符合要求的variable列表
        boolean withSpefcifiedMethod = true;
        String specifiedMethodName = this.parameters.getMethodName();// 获取指定的目标Method名字
        if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {// 如果未指定待注入的Method
            withSpefcifiedMethod = false;// 记录标记
        }
        int length = allMethods.size();
        for (int i = 0; i < length; i++) {
            SootMethod method = allMethods.get(i);
            List<SootClass> declaredExcepts = method.getExceptions();
            Body tmpBody;
            try {
                tmpBody = method.retrieveActiveBody();
            } catch (Exception e) {
                // currently we don't know how to deal with this case
                logger.info("Retrieve Body failed!");
                continue;
            }
            if (tmpBody == null) {
                continue;
            }
            Chain<Trap> traps = tmpBody.getTraps();
            if ((declaredExcepts.size() == 0) && (traps.size() == 0)) {
                continue;
            }
            if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>")) && (!method.getName().contains("<clinit>"))) {
                allQualifiedMethods.add(method);
            } else {
                // it's strict, only when the method satisfies the condition and with the
                // specified name
                if (method.getName().equals(specifiedMethodName)) {// method names are strictly compared
                    allQualifiedMethods.add(method);
                }
            }
        }

        this.allQualifiedMethods = allQualifiedMethods;
    }

    // 执行注入操作
    private boolean inject(Body b, String scope) {
        try {
            if (scope.equals("throw")) {// 对throw阶段进行故障注入
                List<SootClass> allExceptions = b.getMethod().getExceptions(); // 获取所有的Exception类型
                while (true) {
                    int exceptionSize = allExceptions.size();
                    if (exceptionSize == 0) { // 若Exception类型数目为0
                        break;
                    }
                    int exceptionIndex = new Random(System.currentTimeMillis()).nextInt(exceptionSize); // 随机选择待注入的Exception类型
                    SootClass targetException = allExceptions.get(exceptionIndex); // 获取目标Exception类型
                    if (this.injectThrowShort(b, targetException)) { // 对目标Exception类型进行故障注入
                        return true;
                    }
                }
            } else if (scope.equals("catch")) {// 对catch阶段进行故障注入
                List<Trap> allTraps = this.getAllTraps(b); // 获取所有的Trap类型
                while (true) {
                    int trapsSize = allTraps.size();
                    if (trapsSize == 0) {// 若Trap类型数目为0
                        break;
                    }
                    int trapIndex = new Random(System.currentTimeMillis()).nextInt(trapsSize); // 随机选择待注入的Trap类型
                    Trap targetTrap = allTraps.get(trapIndex);// 获取目标Trap类型
                    allTraps.remove(trapIndex);
                    if (this.injectTryShort(b, targetTrap)) {// 对目标Trap类型进行故障注入
                        return true;
                    }
                }
            }
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
            return false;
        }
        return false;
    }

    // 获取全部的Trap类型
    private List<Trap> getAllTraps(Body b) {
        List<Trap> allTraps = new ArrayList<Trap>();
        Iterator<Trap> trapItr = b.getTraps().snapshotIterator(); // 获取Trap快照迭代器
        while (trapItr.hasNext()) {
            allTraps.add(trapItr.next()); // 记录Trap
        }
        return allTraps; // 返回全部的Trap
    }

    // 注入TryShort类故障
    private boolean injectTryShort(Body b, Trap trap) {
        // directly throw the exception at the beginning of the trap
        Chain<Unit> units = b.getUnits();// 获取当前代码语句

        Unit beginUnit = trap.getBeginUnit();//获取Trap的开始语句
        List<Stmt> stmts = this.getShortTryStatements(b, trap);// 获取TryShort故障注入的代码语句
        for (int i = 0; i < stmts.size(); i++) {
            if (i == 0) {
                // here we add the exception before the first statement if this block
                units.insertBefore(stmts.get(i), beginUnit);// 首句在Trap开始语句前插入
            } else {
                units.insertAfter(stmts.get(i), stmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), stmts.get(0));// 首句在stmts[0]前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, beginUnit);// 获取条件语句
        for (int i = 0; i < conditionStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));// 首句在preStmts[-1]后插入
            } else {
                units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }

        List<Stmt> actStmts = this.createActivateStatement(b);// 获取激活操作语句
        for (int i = 0; i < actStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));// 首句在conditionStmts[-1]后插入
            } else {
                units.insertAfter(actStmts.get(i), actStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        return true;

    }

    // 获取TryShort类故障注入语句
    private List<Stmt> getShortTryStatements(Body b, Trap trap) {
        // Decide the exception to be thrown
        SootClass exception = trap.getException(); // 获取exception类
        this.injectInfo.put("VariableName", exception.getName());
        // Find the constructor of this Exception
        SootMethod constructor = null;
        SootMethod constructor2 = null;
        Iterator<SootMethod> smIt = exception.getMethods().iterator(); // 获取exception类中method的迭代器
        while (smIt.hasNext()) {
            SootMethod tmp = smIt.next(); // 获取exception类中method
            String signature = tmp.getSignature(); // 获取method的签名
            // we should also decide which constructor should be used
            if (signature.contains("void <init>(java.lang.String)")) { // 如果函数签名为"void <init>(java.lang.String)"，则记录该构造函数
                constructor = tmp;
            }
            if (signature.contains("void <init>()")) {// 如果函数签名为"void <init>()"，则记录该构造函数
                constructor2 = tmp;
            }
        }
        if ((constructor == null) && (constructor2 == null)) { // 若两个构造函数均不存在，则报错
            // this is a fatal error
            logger.error("Failed to find a constructor for this exception");
            return null;
        }
        // create a exception and initialize it
        if (constructor != null) { // 若利用String的构造函数不为空
            Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception)); // 创建局部变量"tmpException"
            b.getLocals().add(lexception); // 声明局部变量"tmpException"
            AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception))); // 赋值语句 tmpException = new exception
            InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception,
                    constructor.makeRef(), StringConstant.v("Fault Injection")));// 调用语句 tmpException = constructor("Fault Injection")
            ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception); // 抛出语句 throw tmpException
            List<Stmt> stmts = new ArrayList<Stmt>();
            stmts.add(assignStmt); // 记录赋值语句 tmpException = new exception
            stmts.add(invStmt); // 记录调用语句 tmpException = constructor("Fault Injection")
            stmts.add(throwStmt); // 记录抛出语句 throw tmpException
            return stmts; // 返回故障注入语句
        } else {
            Local lexception = Jimple.v().newLocal("tmpException", RefType.v(exception));// 创建局部变量"tmpException"
            b.getLocals().add(lexception);// 声明局部变量"tmpException"
            AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception, Jimple.v().newNewExpr(RefType.v(exception)));// 赋值语句 tmpException = new exception
            InvokeStmt invStmt = Jimple.v()
                    .newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor2.makeRef()));// 调用语句 tmpException = constructor2()
            ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);// 抛出语句 throw tmpException
            List<Stmt> stmts = new ArrayList<Stmt>();
            stmts.add(assignStmt);// 记录赋值语句 tmpException = new exception
            stmts.add(invStmt); // 记录调用语句 tmpException = constructor2()
            stmts.add(throwStmt);// 记录抛出语句 throw tmpException
            return stmts;// 返回故障注入语句
        }

    }

    // 注入ThrowShort类故障
    private boolean injectThrowShort(Body b, SootClass exception) {
        Chain<Unit> units = b.getUnits();// 获取当前代码语句

        this.injectInfo.put("VariableName", exception.getName());
        Unit firstUnit = units.getFirst(); // 获取Method的开始语句
        List<Stmt> stmts = this.getShortThrowStatements(b, exception);// 获取ThrowShort故障注入的代码语句
        for (int i = 0; i < stmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(stmts.get(i), firstUnit);// 首句在firstUnit前插入
            } else {
                units.insertAfter(stmts.get(i), stmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), stmts.get(0));// 首句在stmts[0]前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, firstUnit);// 获取条件语句，根据激活模式或条件判断是否激活故障
        for (int i = 0; i < conditionStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));// 首句在preStmts[-1]后插入
            } else {
                units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }

        List<Stmt> actStmts = this.createActivateStatement(b);// 获取故障注入后的操作语句，记录日志等
        for (int i = 0; i < actStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));// 首句在conditionStmts[-1]后插入
            } else {
                units.insertAfter(actStmts.get(i), actStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        return true;

    }

    // 获取ThrowShort类故障注入语句
    private List<Stmt> getShortThrowStatements(Body b, SootClass exception) {
        // Find the constructor of this Exception
        SootMethod constructor = null;
        Iterator<SootMethod> smIt = exception.getMethods().iterator(); // 获取exception类中method的迭代器
        while (smIt.hasNext()) {
            SootMethod tmp = smIt.next();// 获取exception类中method
            String signature = tmp.getSignature(); // 获取method的签名
            // we should also decide which constructor should be used
            if (signature.contains("void <init>(java.lang.String)")) {// 如果函数签名为"void <init>(java.lang.String)"，则记录该构造函数
                constructor = tmp;
            }
        }

        // create a exception and initialize it
        Local tmpException = Jimple.v().newLocal("tmpSootException", RefType.v(exception)); // 创建局部变量"tmpSootException"
        b.getLocals().add(tmpException);// 声明局部变量"tmpSootException"
        AssignStmt assignStmt = Jimple.v().newAssignStmt(tmpException, Jimple.v().newNewExpr(RefType.v(exception))); // 赋值语句 tmpSootException = new exception
        InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(tmpException,
                constructor.makeRef(), StringConstant.v("Fault Injection")));// 调用语句 tmpSootException = constructor("Fault Injection")

        // Throw it directly
        ThrowStmt throwStmt = Jimple.v().newThrowStmt(tmpException);// 抛出语句 throw tmpSootException

        List<Stmt> stmts = new ArrayList<Stmt>();
        stmts.add(assignStmt); // 记录赋值语句 tmpSootException = new exception
        stmts.add(invStmt);// 记录调用语句 tmpSootException = constructor("Fault Injection")
        stmts.add(throwStmt);// 记录抛出语句 throw tmpSootException
        return stmts;// 返回故障注入语句
    }

    // 获取待注入variable的类别
    private List<String> getTargetScope(String variableScope) {
        List<String> scopes = new ArrayList<String>();// 声明待注入域集合列表
        if ((variableScope == null) || (variableScope == "")) {// 若未指定待注入域
            scopes.add("throw");// 待注入域集合添加"throw"类
            scopes.add("catch");// 待注入域集合添加"catch"类
        } else {// 若指定待注入域
            scopes.add(variableScope);// 直接在注入域集合添加指定的域类型
        }
        return scopes;
    }

}
