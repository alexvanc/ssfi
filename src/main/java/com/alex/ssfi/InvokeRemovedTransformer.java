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
import soot.Unit;
import soot.jimple.GotoStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.util.Chain;

public class InvokeRemovedTransformer extends BasicTransformer {
    private final Logger logger = LogManager.getLogger(InvokeRemovedTransformer.class);

    public InvokeRemovedTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "UNUSED_INVOKE_REMOVED_FAULT");// 标记故障类型为，"UNUSED_INVOKE_REMOVED_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

        logger.debug("Try to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // finally perform injection
        if (this.inject(b)) {// 进行注入
            this.parameters.setInjected(true);// 标记注入成功
            this.recordInjectionInfo();// 记录注入信息
            logger.debug("Succeed to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
            return;
        } else {
            logger.debug("Failed injection:+ " + this.formatInjectionInfo());
        }

        logger.debug("Fail to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

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
    private void initAllQualifiedMethods(Body b) {// 初始化符合要求的方法
        List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();// 获取当前body能够获取的全部Method
        List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();// 声明符合要求的Method列表
        boolean withSpefcifiedMethod = true;
        String specifiedMethodName = this.parameters.getMethodName();// 获取指定的目标Method名字
        if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {// 如果未指定待注入的Method
            withSpefcifiedMethod = false;// 记录标记
        }
        int length = allMethods.size();
        for (int i = 0; i < length; i++) {
            SootMethod method = allMethods.get(i);

            if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>")) && (!method.getName().contains("<clinit>"))) {
                //methods like <init> are used during verification perios, shouldn't be injected
                allQualifiedMethods.add(method);
                break;
            } else {
                // it's strict, only when the method satisfies the condition and with the
                // specified name
                if (method.getName().equals(specifiedMethodName)) {// method names are strictly compared
                    allQualifiedMethods.add(method);
                    break;
                }
            }
        }

        this.allQualifiedMethods = allQualifiedMethods;
    }

    // 执行注入操作
    private boolean inject(Body b) {
        try {
            List<Stmt> allInvokeStmts = this.getAllInvokeStmts(b); // 获取全部Invoke语句
            while (true) {
                int invokeStmtSize = allInvokeStmts.size();
                if (invokeStmtSize == 0) {// 若待注入的Invoke语句数量为0，则返回上级程序
                    break;
                }
                int stmtIndex = new Random(System.currentTimeMillis()).nextInt(invokeStmtSize); //随机选择注入点的索引
                Stmt targetInvokeStmt = allInvokeStmts.get(stmtIndex);//获取目标语句
                allInvokeStmts.remove(stmtIndex);
                if (this.injectInvokeRemoval(b, targetInvokeStmt)) {// 注入InvokeRemoval故障
                    return true;
                }
            }

        } catch (Exception e) {
            logger.error(e.getMessage());
            return false;
        }
        return false;
    }

    // 注入InvokeRemovalt故障
    private boolean injectInvokeRemoval(Body b, Stmt targetStmt) {
        Chain<Unit> units = b.getUnits();
        Unit nextUnit = units.getSuccOf(targetStmt);//获取目标语句的下一句

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), targetStmt);// 首句在targetStmt前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, targetStmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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

        GotoStmt skipInvoke = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过Invoke语句
        units.insertAfter(skipInvoke, actStmts.get(actStmts.size() - 1));// 在actStmts[-1]后插入

        return true;
    }

    // 获取全部Invoke语句
    private List<Stmt> getAllInvokeStmts(Body b) {
        List<Stmt> allInvokeStmts = new ArrayList<Stmt>();
        Iterator<Unit> unitItr = b.getUnits().snapshotIterator();// 获取b内的语句快照迭代器
        while (unitItr.hasNext()) {
            Unit tmpUnit = unitItr.next();
            if (tmpUnit instanceof InvokeStmt) {// 若该语句为Switch语句，记录该Invoke语句
                allInvokeStmts.add((Stmt) tmpUnit);
            }
        }
        return allInvokeStmts;
    }

}
