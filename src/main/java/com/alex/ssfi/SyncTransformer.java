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
import soot.Modifier;
import soot.SootMethod;
import soot.Trap;
import soot.Unit;
import soot.jimple.EnterMonitorStmt;
import soot.jimple.ExitMonitorStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.Stmt;
import soot.jimple.ThrowStmt;
import soot.util.Chain;

public class SyncTransformer extends BasicTransformer {
    private final Logger logger = LogManager.getLogger(SyncTransformer.class);

    public SyncTransformer(RunningParameter parameter) {
        this.parameters = parameter;
    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {// 核心的回调方法
        while (!this.parameters.isInjected()) {
            // in this way, all the FIs are performed in the first function of this class
            SootMethod targetMethod = this.generateTargetMethod(b);// 获取待注入的Method
            if (targetMethod == null) {// 若获取不到待注入的Method，则返回上级程序
                return;
            }
            Body tmpBody;
            try {
                tmpBody = targetMethod.retrieveActiveBody();// 获取Method对应的Body
            } catch (Exception e) {// 若获取Method对应的Body失败，则跳过本次Method，并尝试获取下一个待注入的Method
                logger.info("Retrieve Active Body Failed!");
                continue;
            }
            if (tmpBody == null) {// 如果获取的Body为空，则跳过本次Method，并尝试获取下一个待注入的Method
                continue;
            }
            this.startToInject(targetMethod.getActiveBody()); // 开始故障注入。对待注入的Method对应的Body进行故障注入
        }
    }

    // 对于目标Body进行故障注入
    private void startToInject(Body b) {
        // no matter this inject fails or succeeds, this targetMethod is already used
        SootMethod targetMethod = b.getMethod();// 获取目标Method
        this.injectInfo.put("FaultType", "SYNC_FAULT");// 标记故障类型为，"SYNC_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名
        this.injectInfo.put("ActivationMode", "always");

        List<String> scopes = getSyncScope(this.parameters.getVariableScope());//获取全部可注入的域

        logger.debug("Try to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // randomly combine scope, type , variable, action
        while (true) {
            int scopesSize = scopes.size();
            if (scopesSize == 0) {//若全部的待注入的域数目为0，则退出循环
                logger.debug("Cannot find qualified scopes");
                break;
            }
            int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize);//随机选取目标域索引
            String scope = scopes.get(scopeIndex);//获取目标域
            scopes.remove(scopeIndex);
            this.injectInfo.put("VariableScope", scope);
            // injection sync fault for methods
            List<String> actions = getPossibleActionForScope(scope, this.parameters.getAction());// 获取所有可行的操作行为，作为待注入的操作集合
            while (true) {
                int actionSize = actions.size();
                if (actionSize == 0) {//若全部的待注入的操作数目为0，则退出循环
                    break;
                }
                int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);// 随机选择待注入的操作的索引
                String action = actions.get(actionIndex);// 获取目标注入操作
                this.injectInfo.put("Action", action);

                // finally perform injection
                if (this.inject(b, scope, action)) {// 进行注入
                    this.parameters.setInjected(true);// 标记注入成功
                    this.recordInjectionInfo(); // 记录注入信息
                    logger.debug("Succeed to inject SYNC_FAULT into " + this.injectInfo.get("Package") + " "
                            + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                    return;
                } else {
                    logger.debug("Failed injection:+ " + this.formatInjectionInfo());
                }
            }
        }

        logger.debug("Fail to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

    }

    // 执行注入操作
    private boolean inject(Body b, String scope, String action) {
        if (scope.equals("method")) {// 若目标域为"method"
            return this.injectMethodWithAction(b, action); // 注入Method

        } else {// block
            return this.injectBlockWithAction(b, action); // 注入Block
        }
    }

    // 给定注入的操作类型，对Block进行Sync故障注入
    private boolean injectBlockWithAction(Body b, String action) {

        Chain<Unit> units = b.getUnits();// 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator();// 获取代码语句的快照迭代器
        boolean started = false;
        while (unitIt.hasNext()) {// delete the first synchronized block
            Stmt tmpStmt = (Stmt) unitIt.next(); // 获取当前语句
            if ((started) && (tmpStmt instanceof IdentityStmt)) { //若tmpStmt不是EnterMonitorStmt，且为IdentityStmt
                Stmt nextStmt = (Stmt) units.getSuccOf(tmpStmt); // 获取下一句语句
                Stmt nextNextStmp = (Stmt) units.getSuccOf(nextStmt); // 获取下下句语句
                if ((nextStmt instanceof ExitMonitorStmt) && (nextNextStmp instanceof ThrowStmt)) { // 下一句语句为ExitMonitorStmt，且下下句语句为ThrowStmt
                    Iterator<Trap> trapsItr = b.getTraps().snapshotIterator(); // 获取Trap快照迭代器
                    while (trapsItr.hasNext()) {
                        Trap tmpTrap = trapsItr.next(); // 获取当前Trap
                        Unit startUnit = tmpTrap.getHandlerUnit(); // 获取Trap对应的HandlerUnit
                        if (startUnit.equals(tmpStmt)) { // 若Trap对应的HandlerUnit为tmpStmt，则移除tmpStmt
                            b.getTraps().remove(tmpTrap);
                        }
                    }
                    units.remove(tmpStmt);// 移除 tmpStmt
                    units.remove(nextStmt); // 移除下句 ExitMonitorStmt
                    units.remove(nextNextStmp); // 移除下下句 ThrowStmt
                    return true;
                }
            } else if ((started) && (tmpStmt instanceof ExitMonitorStmt)) {//若tmpStmt不是EnterMonitorStmt，且为ExitMonitorStmt
                units.remove(tmpStmt); // 移除 tmpStmt

            }
            if (tmpStmt instanceof EnterMonitorStmt) {
                logger.info(this.formatInjectionInfo());
                units.remove(tmpStmt); // 移除 tmpStmt
                started = true; // 记录已经找到过开始语句
            }

        }
        return false;
    }

    // 给定注入的操作类型，对Method进行Sync故障注入
    private boolean injectMethodWithAction(Body b, String action) {
        SootMethod targetMethod = b.getMethod();
        if (targetMethod.isSynchronized()) {
            if (action.equals("ASYNC")) {// 若注入操作为异步
                logger.info(this.formatInjectionInfo());
                int originalModifiers = targetMethod.getModifiers();
                int targetModifiers = originalModifiers & (~Modifier.SYNCHRONIZED);// 更改目标修饰符为异步
                targetMethod.setModifiers(targetModifiers);
                System.out.println(targetMethod.isStatic());
                return true;
            }
        } else {
            if (action.equals("SYNC")) { // 若注入操作为同步
                logger.info(this.formatInjectionInfo());
                int originalModifiers = targetMethod.getModifiers();
                int targetModifiers = originalModifiers | (Modifier.SYNCHRONIZED); // 更改目标修饰符为同步
                targetMethod.setModifiers(targetModifiers);
                System.out.println(targetMethod.isStatic());
                return true;
            }
        }
        return false;
    }

    // 获取待注入的同步域
    private List<String> getSyncScope(String variableScope) {
        List<String> scopes = new ArrayList<String>();// 声明待注入域集合列表
        if ((variableScope == null) || (variableScope == "")) {// 若未指定待注入域
            scopes.add("method"); // 待注入域集合添加"method"类
            scopes.add("block");// 待注入域集合添加"block"类
        } else {// 若指定了待注入域
            scopes.add(variableScope);// 直接在注入域集合添加指定的域类型
        }
        return scopes;
    }

    // 对于给定的待注入scope，获取所有可能的注入操作类型
    private List<String> getPossibleActionForScope(String scope, String speficifiedAction) {
        // For simplicity, we don't check whether the action is applicable to this type
        // of variables
        // this should be done in the configuration validator
        List<String> possibleActions = new ArrayList<String>();
        String specifiedAction = this.parameters.getAction();
        if ((specifiedAction == null) || (specifiedAction == "")) {
            if (scope.equals("method")) {// another scope is block
                possibleActions.add("SYNC");// 可能的注入操作为"SYNC"
                possibleActions.add("ASYNC");// 可能的注入操作为"ASYNC"
            } else {
                possibleActions.add("ASYNC");// 可能的注入操作为"ASYNC"
            }
        } else {
            // here we believe users know how to specify right configurations
            possibleActions.add(specifiedAction);// 可能的注入操作为指定的操作
        }
        return possibleActions;
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
        List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();// 声明符合要求的Method列表
        boolean withSpefcifiedMethod = true;
        String specifiedMethodName = this.parameters.getMethodName();// 获取指定的目标Method名字
        if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {// 如果未指定待注入的Method
            withSpefcifiedMethod = false;// 记录标记
        }
        int length = allMethods.size();
        for (int i = 0; i < length; i++) {
            SootMethod method = allMethods.get(i);

            if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>"))
                    && (!method.getName().contains("<clinit>"))) {
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
