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
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.GotoStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.util.Chain;

//this type of fault only applies to try-catch block
public class ExceptionUnHandledTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(ValueTransformer.class);

    public ExceptionUnHandledTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "EXCEPTION_UNHANDLED_FAULT");// 标记故障类型为，"EXCEPTION_UNHANDLED_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

        logger.debug("Try to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // finally perform injection
        if (this.inject(b)) {// 进行注入
            this.parameters.setInjected(true);// 标记注入成功
            this.recordInjectionInfo();// 记录注入信息
            logger.debug("Succeed to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
            return;
        } else {
            logger.debug("Failed injection:+ " + this.formatInjectionInfo());
        }

        logger.debug("Fail to inject EXCEPTION_UNHANDLED_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

    }

    // 执行注入操作
    private boolean inject(Body b) {
        try {
            List<Trap> allTraps = this.getAllTraps(b);// 获取所有的Trap类型
            while (true) {
                int trapSize = allTraps.size();
                if (trapSize == 0) {// 若Trap类型数目为0
                    return false;
                }
                int trapIndex = new Random(System.currentTimeMillis()).nextInt(trapSize);// 随机选择待注入的Trap类型
                Trap targetTrap = allTraps.get(trapIndex);// 获取目标Trap类型
                allTraps.remove(trapIndex);
                if (this.injectUnhandled(b, targetTrap)) {// 对目标Trap类型进行故障注入
                    return true;
                }
            }
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
            return false;
        }
    }

    // 注入Unhandled类故障
    private boolean injectUnhandled(Body b, Trap trap) {
        Chain<Unit> units = b.getUnits();// 获取当前代码语句

        Unit beginUnit = trap.getHandlerUnit();//获取Trap的处理语句
        Unit afterBeginUnit = units.getSuccOf(beginUnit);//获取处理语句的后续语句
        Unit endUnit = trap.getEndUnit();//获取Trap的结束语句

        // all the process steps after this @caughtexception stmt

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(preStmts.get(i), beginUnit);// 首句在stmts[0]前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, afterBeginUnit);// 获取条件语句
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
        if (endUnit instanceof GotoStmt) { // 若endUnit是goto语句
            GotoStmt gTryEndUnit = (GotoStmt) trap.getEndUnit();
            Unit outCatchUnit = gTryEndUnit.getTarget(); // 获取goto语句的目标
            GotoStmt skipProcess = Jimple.v().newGotoStmt(outCatchUnit); // 构造goto语句
            units.insertAfter(skipProcess, actStmts.get(actStmts.size() - 1)); // 在actStmts[-1]后插入
        } else {// 若endUnit是return语句
            // return or return void
            Stmt returnStmt = (Stmt) endUnit.clone();
            units.insertAfter(returnStmt, actStmts.get(actStmts.size() - 1));// 在actStmts[-1]后插入
        }
        return true;
    }

    // 获取全部的Trap类型
    private List<Trap> getAllTraps(Body b) {
        List<Trap> allTraps = new ArrayList<Trap>();
        Iterator<Trap> trapItr = b.getTraps().snapshotIterator();// 获取Trap快照迭代器
        while (trapItr.hasNext()) {
            allTraps.add(trapItr.next());// 记录Trap
        }
        return allTraps;// 返回全部的Trap
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
            if (traps.size() == 0) {
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
