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
import soot.Scene;
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

public class ExceptionUncaughtTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(ExceptionUncaughtTransformer.class);

    public ExceptionUncaughtTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "EXCEPTION_UNCAUGHT_FAULT");// 标记故障类型为，"EXCEPTION_UNCAUGHT_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

        logger.debug("Try to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // finally perform injection
        if (this.inject(b)) {// 进行注入
            this.parameters.setInjected(true);// 标记注入成功
            this.recordInjectionInfo();// 记录注入信息
            logger.debug("Succeed to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
            return;
        } else {
            logger.debug("Failed injection:+ " + this.formatInjectionInfo());
        }

        logger.debug("Fail to inject EXCEPTION_UNCAUGHT_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

    }

    // 执行注入操作
    private boolean inject(Body b) {
        // we directly throw a Exception instance at the entrance of the method
        // we don't distinguish try-catch, throw, none of the two anymore
        try {
            Chain<Unit> units = b.getUnits();// 获取当前代码语句
            List<Stmt> stmts = this.getUncaughtStatments(b); // 获取Uncaught故障注入的代码语句
            Unit firstUnit = units.getFirst();// 获取Method的开始语句
            for (int i = 0; i < stmts.size(); i++) {
                if (i == 0) {
                    // here we add the exception before the first statement of this method
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
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
            return false;
        }
    }

    // 获取Uncaught类故障注入语句
    private List<Stmt> getUncaughtStatments(Body b) {
        // Directly throw a new Exception instance
        SootClass rootExceptionClass = Scene.v().getSootClass("java.lang.Exception");// 获取root exception类
        // Find the constructor of this Exception
        SootMethod constructor = null;
        Iterator<SootMethod> smIt = rootExceptionClass.getMethods().iterator();// 获取exception类中method的迭代器
        while (smIt.hasNext()) {
            SootMethod tmp = smIt.next();// 获取exception类中method
            String signature = tmp.getSignature();// 获取method的签名
            // TO-DO
            // we should also decide which constructor should be used
            if (signature.contains("void <init>(java.lang.String)")) {// 如果函数签名为"void <init>(java.lang.String)"，则记录该构造函数
                constructor = tmp;
            }
        }
        // create an exception and initialize it
        Local lexception = Jimple.v().newLocal("tmpException", RefType.v(rootExceptionClass));// 创建局部变量"tmpException"
        b.getLocals().add(lexception);
        AssignStmt assignStmt = Jimple.v().newAssignStmt(lexception,
                Jimple.v().newNewExpr(RefType.v(rootExceptionClass)));// 赋值语句 tmpException = new exception
        InvokeStmt invStmt = Jimple.v().newInvokeStmt(Jimple.v().newSpecialInvokeExpr(lexception, constructor.makeRef(),
                StringConstant.v("Fault Injection")));// 调用语句 tmpException = constructor("Fault Injection")

        ThrowStmt throwStmt = Jimple.v().newThrowStmt(lexception);// 抛出语句 throw tmpException
        List<Stmt> stmts = new ArrayList<Stmt>();
        stmts.add(assignStmt);// 记录赋值语句 tmpException = new exception
        stmts.add(invStmt);// 记录调用语句 tmpException = constructor("Fault Injection")
        stmts.add(throwStmt);// 记录抛出语句 throw tmpException
        return stmts;// 返回故障注入语句
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

    // for this fault type,we simply assume all methods satisfy the condition
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
            boolean noExceptionDeclared = true;
            boolean noExceptionCaught = true;
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
            Iterator<Trap> caughtExcepts = tmpBody.getTraps().snapshotIterator();
            for (int j = 0, size = declaredExcepts.size(); j < size; j++) {
                SootClass exception = declaredExcepts.get(j);
                if (exception.getName().equals("java.lang.Exception")) {
                    noExceptionDeclared = false;
                    break;
                }
            }
            while (caughtExcepts.hasNext()) {
                Trap trap = caughtExcepts.next();
                // actually it's not that accurate
                // multiple try-catch blocks and multiple catch blocks for one try-catch is the
                // same in soot
                SootClass exception = trap.getException();
                if (exception.getName().equals("java.lang.Exception")) {
                    noExceptionDeclared = false;
                    break;
                }
            }
            if ((!noExceptionDeclared) || (!noExceptionCaught)) {
                // Exception are either declared or caught in this method
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

}
