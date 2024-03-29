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
import soot.jimple.Expr;
import soot.jimple.GeExpr;
import soot.jimple.GotoStmt;
import soot.jimple.GtExpr;
import soot.jimple.IfStmt;
import soot.jimple.Jimple;
import soot.jimple.LeExpr;
import soot.jimple.LtExpr;
import soot.jimple.Stmt;
import soot.util.Chain;

public class ConditionBorderTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(ConditionBorderTransformer.class);

    public ConditionBorderTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "CONDITION_BORDER_FAULT");// 标记故障类型为，"CONDITION_BORDER_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

        logger.debug("Try to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // actually we should do the randomization in the inject method
        // to keep coding stype the same
        List<Stmt> allCompareStmt = getAllCompareStmt(b);//获取全部Compare语句
        while (true) {
            int compStmtSize = allCompareStmt.size();
            if (compStmtSize == 0) {//若待注入的Compare语句数量为0，则返回上级程序
                break;
            }
            int randomCompIndex = new Random(System.currentTimeMillis()).nextInt(compStmtSize); // 随机选取目标Compare语句的索引
            Stmt targetCompStmt = allCompareStmt.get(randomCompIndex);// 获取目标Compare语句
            allCompareStmt.remove(randomCompIndex);

            // finally perform injection
            if (this.inject(b, targetCompStmt)) {// 进行注入
                this.parameters.setInjected(true);// 标记注入成功
                this.recordInjectionInfo();// 记录注入信息
                logger.debug("Succeed to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
                        + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                return;
            } else {
                logger.debug("Failed injection:+ " + this.formatInjectionInfo());
            }
        }

        logger.debug("Fail to inject CONDITION_BORDER_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

    }

    // 获取全部Compare语句
    private List<Stmt> getAllCompareStmt(Body b) {
        List<Stmt> allCompareStmt = new ArrayList<Stmt>();
        Chain<Unit> units = b.getUnits();
        Iterator<Unit> unitItr = units.iterator();// 获取b内的语句快照迭代器
        // choose one scope to inject faults
        // if not specified, then randomly choose
        while (unitItr.hasNext()) {
            Stmt stmt = (Stmt) unitItr.next();
            if (stmt instanceof IfStmt) {// 若该语句为IF语句，记录该IF语句
                allCompareStmt.add(stmt);

            }
        }
        return allCompareStmt;

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
            Iterator<Unit> unitItr = tmpBody.getUnits().snapshotIterator();
            while (unitItr.hasNext()) {
                Unit tmpUnit = unitItr.next();
                if (tmpUnit instanceof IfStmt) {
                    // we only deal with <, >, <=, >=
                    IfStmt tmpIfStmt = (IfStmt) tmpUnit;
                    Expr expr = (Expr) tmpIfStmt.getCondition();
                    if ((expr instanceof GtExpr) || (expr instanceof GeExpr) || (expr instanceof LtExpr)
                            || (expr instanceof LeExpr)) {
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

        }

        this.allQualifiedMethods = allQualifiedMethods;
    }

    // 执行注入操作（注入ConditionBorder故障）
    private boolean inject(Body b, Stmt stmt) {
        Chain<Unit> units = b.getUnits();
        try {
            IfStmt ifStmt = (IfStmt) stmt;
            IfStmt newIfStmt = (IfStmt) ifStmt.clone();
            Expr expr = (Expr) newIfStmt.getConditionBox().getValue(); //获取IF语句中的表达式
            Unit nextUnit = units.getSuccOf(ifStmt);//获取if语句之后的语句
            if (expr instanceof GtExpr) {//若expr为>表达式

                GtExpr gtExp = (GtExpr) expr;
                GeExpr geExp = Jimple.v().newGeExpr(gtExp.getOp1(), gtExp.getOp2());//创建Ge表达式
                newIfStmt.setCondition(geExp);//设置新的IF语句条件
                units.insertBefore(newIfStmt, ifStmt);// 在原ifStmt前插入新的IF语句

                List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
                for (int i = 0; i < preStmts.size(); i++) {
                    if (i == 0) {
                        units.insertBefore(preStmts.get(i), newIfStmt);// 首句在newIfStmt前插入
                    } else {
                        units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
                    }
                }
                List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
                GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过原IF语句
                units.insertAfter(skipIfStmt, newIfStmt);
                return true;
            } else if (expr instanceof GeExpr) {//若expr为>=表达式
                GeExpr geExp = (GeExpr) expr;
                GtExpr gtExp = Jimple.v().newGtExpr(geExp.getOp1(), geExp.getOp2());//创建Gt表达式
                newIfStmt.setCondition(gtExp);//设置新的IF语句条件
                units.insertBefore(newIfStmt, ifStmt);// 在原ifStmt前插入新的IF语句

                List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
                for (int i = 0; i < preStmts.size(); i++) {
                    if (i == 0) {
                        units.insertBefore(preStmts.get(i), newIfStmt);// 首句在newIfStmt前插入
                    } else {
                        units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
                    }
                }
                List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
                GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过原IF语句
                units.insertAfter(skipIfStmt, newIfStmt);
                return true;
            } else if (expr instanceof LtExpr) {//若expr为<表达式
                LtExpr ltExp = (LtExpr) expr;
                LeExpr leExp = Jimple.v().newLeExpr(ltExp.getOp1(), ltExp.getOp2());//创建Le表达式
                newIfStmt.setCondition(leExp);//设置新的IF语句条件
                units.insertBefore(newIfStmt, ifStmt);// 在原ifStmt前插入新的IF语句

                List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
                for (int i = 0; i < preStmts.size(); i++) {
                    if (i == 0) {
                        units.insertBefore(preStmts.get(i), newIfStmt);// 首句在newIfStmt前插入
                    } else {
                        units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
                    }
                }
                List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
                GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过原IF语句
                units.insertAfter(skipIfStmt, newIfStmt);
                return true;
            } else if (expr instanceof LeExpr) {//若expr为<=表达式
                LeExpr leExp = (LeExpr) expr;
                LtExpr ltExp = Jimple.v().newLtExpr(leExp.getOp1(), leExp.getOp2());//创建Lt表达式
                newIfStmt.setCondition(ltExp);//设置新的IF语句条件
                units.insertBefore(newIfStmt, ifStmt);// 在原ifStmt前插入新的IF语句

                List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
                for (int i = 0; i < preStmts.size(); i++) {
                    if (i == 0) {
                        units.insertBefore(preStmts.get(i), newIfStmt);// 首句在newIfStmt前插入
                    } else {
                        units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
                    }
                }
                List<Stmt> conditionStmts = this.getConditionStmt(b, stmt);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
                GotoStmt skipIfStmt = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过原IF语句
                units.insertAfter(skipIfStmt, newIfStmt);
                return true;
            }
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
        }
        return false;
    }

}
