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

public class SwitchFallThroughTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(SwitchFallThroughTransformer.class);

    public SwitchFallThroughTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "SWITCH_FALLTHROUGH_FAULT");// 标记故障类型为，"SWITCH_FALLTHROUGH_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名
        List<String> actions = this.getTargetAction(this.parameters.getAction());// 获取目标操作集合

        logger.debug("Try to inject SWITCH_FALLTHROUGH_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // randomly combine action
        while (true) {
            int actionSize = actions.size();// 获取待注入的操作集合的大小
            if (actionSize == 0) {
                logger.debug("Cannot find qualified actions");
                break;
            }
            int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);// 随机选择待注入的操作的索引
            String action = actions.get(actionIndex);// 获取待注入操作
            actions.remove(actionIndex);// 在待注入的操作的集合中，移除将要进行尝试的操作
            this.injectInfo.put("Action", action);
            List<Stmt> allSwtichStmts = this.getAllSwitchStmt(b);// 获取全部的Switch语句
            while (true) {
                int stmtsSize = allSwtichStmts.size();
                if (stmtsSize == 0) {// 若Switch语句数量为0
                    break;
                }
                int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtsSize);// 随机选择待注入Switch语句的索引
                Stmt switchStmt = allSwtichStmts.get(stmtIndex);//获取目标Switch语句
                allSwtichStmts.remove(stmtIndex);// 在待注入的Switch集合中，移除将要注入的Switch

                // finally perform injection
                if (this.inject(b, action, switchStmt)) {// 进行注入
                    this.parameters.setInjected(true);// 标记注入成功
                    this.recordInjectionInfo();// 记录注入信息
                    logger.debug("Succeed to inject SWITCH_FALLTHROUGH_FAULT into " + this.injectInfo.get("Package")
                            + " " + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                    return;
                } else {
                    logger.debug("Failed injection:+ " + this.formatInjectionInfo());
                }
            }

            logger.debug("Fail to inject" + this.formatInjectionInfo());
        }

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

    // 获取目标操作
    private List<String> getTargetAction(String action) {
        List<String> actions = new ArrayList<String>();
        if ((action == null) || (action == "")) { // 若目标操作为空
            actions.add("break"); // 增加break操作
            actions.add("fallthrough"); // 增加fallthrough操作
        } else { // 若指定目标操作
            actions.add(action); // 直接在待注入操作中添加指定操作
        }
        return actions;
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
                //currently we don't know how to deal with this case
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
    private boolean inject(Body b, String action, Stmt stmt) {
        if (action.equals("break")) {// 若操作类型是break
            try {
                SwitchStmt switchStmt = (SwitchStmt) stmt;
                List<Unit> targets = switchStmt.getTargets();
                List<Integer> availableTargetIndex = new ArrayList<Integer>();

                // first we find the default target for the switch cases
                // the getDefaultTarget() method in soot is not what we want
                // meanwhile we find the cases can be use for injecting a break
                boolean foundBreakTarget = false;
                Unit breakStmt = null;
                Chain<Unit> units = b.getUnits();// 获取当前代码语句
                for (int j = 1; j < targets.size(); j++) {
                    Unit caseBranch = targets.get(j);
                    Unit stmtBeforeCase = units.getPredOf(caseBranch);
                    if (stmtBeforeCase instanceof GotoStmt) {// 若case前的语句是goto语句，则找到break的目标
                        foundBreakTarget = true;
                        breakStmt = stmtBeforeCase;
                    } else {
                        // find all the branches without a break stmt separating itself and next branch
                        availableTargetIndex.add(j - 1);
                    }
                }

                if (!foundBreakTarget) {// couldn't find destination for any break
                    this.logger.debug(this.formatInjectionInfo());
                    return false;
                }
                while (true) {
                    int availableIndexSize = availableTargetIndex.size();// 获取可能的注入点数量
                    if (availableIndexSize == 0) {
                        return false;
                    }
                    int targetIndex = new Random(System.currentTimeMillis()).nextInt(availableIndexSize);//随机选择注入点的索引
//					Unit currentTargetUnit = targets.get(availableTargetIndex.get(targetIndex));
                    Unit nextTargetUnit = targets.get(availableTargetIndex.get(targetIndex) + 1);//获取目标语句
                    availableTargetIndex.remove(targetIndex);
                    if (this.injectSwitchBreak(b, nextTargetUnit, breakStmt)) {// 注入SwitchBreak故障
                        return true;
                    }
                }
            } catch (Exception e) {
                logger.error(e.getMessage());
                logger.error(this.injectInfo.toString());
                return false;
            }
        } else if (action.equals("fallthrough")) {// 若操作类型是fallthrough
            try {
                SwitchStmt switchStmt = (SwitchStmt) stmt;
                List<Unit> targets = switchStmt.getTargets();
                List<Integer> availableTargetIndex = new ArrayList<Integer>();

                // first we find all the branches with a break stmt separating itself with next
                // branch
                Chain<Unit> units = b.getUnits();
                for (int j = 1; j < targets.size(); j++) {
                    Unit caseBranch = targets.get(j);
                    Unit stmtBeforeCase = units.getPredOf(caseBranch);
                    if (stmtBeforeCase instanceof GotoStmt) {// 若case前的语句是goto语句，则记录可注入的case点
                        availableTargetIndex.add(j - 1);
                    }
                }

                while (true) {
                    int availableIndexSize = availableTargetIndex.size();;// 获取可能的注入点数量
                    if (availableIndexSize == 0) {
                        return false;
                    }
                    int targetIndex = new Random(System.currentTimeMillis()).nextInt(availableIndexSize);//随机选择注入点的索引
//					Unit currentTargetUnit = targets.get(availableTargetIndex.get(targetIndex));
                    Unit nextTargetUnit = targets.get(availableTargetIndex.get(targetIndex) + 1);//获取目标语句
                    availableTargetIndex.remove(targetIndex);
                    if (this.injectSwitchFallthrough(b, nextTargetUnit)) {// 注入SwitchFallthrough故障
                        return true;
                    }
                }
            } catch (Exception e) {
                logger.error(e.getMessage());
                logger.error(this.injectInfo.toString());
                return false;
            }
        }
        return false;
    }

    // delete the break stmt which separates two different cases
    private boolean injectSwitchFallthrough(Body b, Unit nextTargetUnit) {
        Chain<Unit> units = b.getUnits();
        Unit breakStmt = units.getPredOf(nextTargetUnit);// 定位到目标break语句

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), breakStmt);// 首句在breakStmt前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, breakStmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
        GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextTargetUnit); // goto语句 delete the break stmt which separates two different cases
        units.insertAfter(skipIfStmt, actStmts.get(actStmts.size() - 1));// 在actStmts[-1]后插入
        return true;

    }

    // add a break stmt between two cases without a break stmt
    private boolean injectSwitchBreak(Body b, Unit nextTargetUnit, Unit breakUnit) {
        Chain<Unit> units = b.getUnits();

        List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), nextTargetUnit);// 首句在nextTargetUnit前插入
            } else {
                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
            }
        }
        List<Stmt> conditionStmts = this.getConditionStmt(b, nextTargetUnit);// 获取条件语句，根据激活模式或条件判断是否激活故障
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

        GotoStmt oriBreakStmt = (GotoStmt) breakUnit;
        GotoStmt breakStmt = Jimple.v().newGotoStmt(oriBreakStmt.getTarget()); // goto语句 add a break stmt between two cases without a break stmt
        units.insertAfter(breakStmt, actStmts.get(actStmts.size() - 1));// 在actStmts[-1]后插入
        return true;
    }

}
