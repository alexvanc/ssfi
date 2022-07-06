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
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.jimple.SwitchStmt;
import soot.util.Chain;

public class SwitchMissDefaultTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(SwitchMissDefaultTransformer.class);

    public SwitchMissDefaultTransformer(RunningParameter parameters) {
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
        SootMethod targetMethod = b.getMethod();// 获取目标Method
        this.injectInfo.put("FaultType", "SWITCH_MISS_DEFAULT_FAULT");// 标记故障类型为，"SWITCH_MISS_DEFAULT_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

        logger.debug("Try to inject SWITCH_MISS_DEFAULT_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        List<Stmt> allSwtichStmts = this.getAllSwitchStmt(b);// 获取全部的Switch语句
        while (true) {
            int stmtsSize = allSwtichStmts.size();
            if (stmtsSize == 0) {// 若待注入的Switch语句数为0，则结束故障注入
                break;
            }
            int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtsSize);// 随机选择待注入的Switch语句的索引
            Stmt switchStmt = allSwtichStmts.get(stmtIndex);//获取待注入Switch语句
            allSwtichStmts.remove(stmtIndex);//在待注入Switch语句集合中，移除将要进行尝试的Switch语句

            // finally perform injection
            if (this.inject(b, switchStmt)) {// 进行注入
                this.parameters.setInjected(true);// 标记注入成功
                this.recordInjectionInfo();// 记录注入信息
                logger.debug("Succeed to inject SWITCH_MISS_DEFAULT_FAULT into " + this.injectInfo.get("Package") + " "
                        + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                return;
            } else {
                logger.debug("Failed injection:+ " + this.formatInjectionInfo());
            }
        }
        logger.debug("Fail to inject" + this.formatInjectionInfo());
    }

    // 获取全部Switch语句
    private List<Stmt> getAllSwitchStmt(Body b) {
        List<Stmt> stmts = new ArrayList<Stmt>();
        Iterator<Unit> unitItr = b.getUnits().snapshotIterator();// 获取b内的语句快照迭代器
        while (unitItr.hasNext()) {
            Stmt tmpStmt = (Stmt) unitItr.next();
            if (tmpStmt instanceof SwitchStmt) { // 若该语句为Switch语句，记录该Switch语句
                stmts.add(tmpStmt);
            }
        }
        return stmts;
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

    // 初始化符合要求的方法
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
            Iterator<Unit> units = tmpBody.getUnits().snapshotIterator();
            while (units.hasNext()) {
                Unit unit = units.next();
                if (unit instanceof SwitchStmt) {
                    if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>")) && (!method.getName().contains("<clinit>"))) {
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
            }
        }
        this.allQualifiedMethods = allQualifiedMethods;
    }

    // 执行注入操作
    private boolean inject(Body b, Stmt stmt) {
        try {
            SwitchStmt switchStmt = (SwitchStmt) stmt;
            List<Unit> targets = switchStmt.getTargets();
            Unit defaultUnit = switchStmt.getDefaultTarget();
            // find a break target to directly go out switch block
            boolean foundBreakTarget = false;
            GotoStmt breakStmt = null;
            Chain<Unit> units = b.getUnits();// 获取当前代码语句
            for (int j = 1; j < targets.size(); j++) {
                Unit caseBranch = targets.get(j);
                Unit stmtBeforeCase = units.getPredOf(caseBranch);
                if (stmtBeforeCase instanceof GotoStmt) {// 若case前的语句是goto语句，则找到break的目标
                    foundBreakTarget = true;
                    breakStmt = (GotoStmt) stmtBeforeCase;
                    break;
                }
            }
            if (!foundBreakTarget) {// couldn't find destination for break
                this.logger.debug(this.formatInjectionInfo());
                return false;
            }
            if (breakStmt.getTarget().equals(defaultUnit)) {// there is no default block in this switch
                this.logger.debug(this.formatInjectionInfo());
                return false;
            }
            if (this.injectMissDefault(b, switchStmt, breakStmt, defaultUnit)) {// 注入MissDefault故障
                return true;
            }
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
        }
        return false;
    }

    // 注入MissDefault故障
    private boolean injectMissDefault(Body b, SwitchStmt switchStmt, GotoStmt breakDestinationStmt,
                                      Unit defaultTargetUnit) {
        Chain<Unit> units = b.getUnits();
        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), defaultTargetUnit);// 首句在defaultTargetUnit前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, defaultTargetUnit);// 获取条件语句，根据激活模式或条件判断是否激活故障
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

        GotoStmt outCatchStmt = Jimple.v().newGotoStmt(breakDestinationStmt.getTarget()); // goto语句 跳过default情况
        units.insertAfter(outCatchStmt, actStmts.get(actStmts.size() - 1));// 在actStmts[-1]后插入
        switchStmt.setDefaultTarget(preStmts.get(0));

        return true;
    }

}
