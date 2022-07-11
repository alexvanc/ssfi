package com.alex.ssfi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.BooleanType;
import soot.ByteType;
import soot.DoubleType;
import soot.FloatType;
import soot.IntType;
import soot.Local;
import soot.LongType;
import soot.Modifier;
import soot.ShortType;
import soot.SootField;
import soot.SootMethod;
import soot.Type;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.VoidType;
import soot.jimple.FieldRef;
import soot.jimple.Jimple;
import soot.jimple.NullConstant;
import soot.jimple.ReturnStmt;
import soot.jimple.Stmt;
import soot.util.Chain;

public class NullTransformer extends BasicTransformer {
    private final Logger logger = LogManager.getLogger(ValueTransformer.class);

    public NullTransformer(RunningParameter parameters) {
        super(parameters);

    }

    public NullTransformer() {
    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {// 核心的回调方法
        while (!this.parameters.isInjected()) {
            SootMethod targetMethod = this.generateTargetMethod(b);// 获取待注入的Method（随机或指定）
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
            this.startToInject(tmpBody);// 开始故障注入。对待注入的Method对应的Body进行故障注入
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

    // for this fault type,we simply assume all methods satisfy the condition
    private void initAllQualifiedMethods(Body b) {
        List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();// 获取当前body能够获取的全部Method
        List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();// 声明符合要求的variable列表
        boolean withSpefcifiedMethod = true;
        String specifiedMethodName = this.parameters.getMethodName(); // 获取指定的目标Method名字
        if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {// 如果未指定待注入的Method
            withSpefcifiedMethod = false;// 记录标记
        }
        int length = allMethods.size();
        for (int i = 0; i < length; i++) {
            SootMethod method = allMethods.get(i);

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

    // 对于目标Body进行故障注入
    private void startToInject(Body b) {
        // no matter this inject fails or succeeds, this targetMethod is already used
        SootMethod targetMethod = b.getMethod();// 获取目标Method
        this.injectInfo.put("FaultType", "NULL_FAULT");// 标记故障类型为，"NULL_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名
        List<String> scopes = getTargetScope(this.parameters.getVariableScope());// 获取目标域集合

        Map<String, List<Object>> allVariables = this.getAllVariables(b);// 获取目标Body相关的所有variable，并按照Field、Local、Parameter分组保存
        logger.debug("Try to inject NULL_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // randomly combine scope, variable
        while (true) {
            int scopesSize = scopes.size();// 获取待注入的域集合
            if (scopesSize == 0) {
                logger.debug("Cannot find qualified scopes");
                break;
            }
            int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize);// 随机选择待注入的域的索引
            String scope = scopes.get(scopeIndex);// 获取待注入的域
            scopes.remove(scopeIndex); // 在待注入的域集合中，移除将要进行尝试的域
            this.injectInfo.put("VariableScope", scope);// 记录待注入的域

            // randomly combine variable, action
            List<Object> qualifiedVariables = this.getQualifiedVariables(b, allVariables, scope);// 获取目标域的variable集合，作为待注入的variable集合
            while (true) {
                int variablesSize = qualifiedVariables.size();// 获取待注入variable集合的大小
                if (variablesSize == 0) {
                    break;
                }
                int variableIndex = new Random(System.currentTimeMillis()).nextInt(variablesSize);// 随机选择待注入variable的索引
                Object variable = qualifiedVariables.get(variableIndex);// 获取待注入的variable
                qualifiedVariables.remove(variableIndex);// 在待注入的variable集合中，移除将要进行尝试的variable

                // finally perform injection
                if (this.inject(b, scope, variable)) {// 进行注入
                    this.parameters.setInjected(true);// 标记注入成功
                    this.recordInjectionInfo();// 记录注入信息
                    logger.debug("Succeed to inject NULL_FAULT into " + this.injectInfo.get("Package") + " "
                            + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                    return;
                } else {
                    logger.debug("Failed injection:+ " + this.formatInjectionInfo());
                }

            }
            logger.debug("Fail to inject NULL_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
        }

    }

    // 执行注入操作
    private boolean inject(Body b, String scope, Object variable) {
        if (scope.equals("local")) {
            return this.injectLocalWithNull(b, (Local) variable);// 对local variable 进行故障注入
        } else if (scope.equals("parameter")) {
            // currently local and parameter are processed in the same way, decide later
            return this.injectParameterWithNull(b, (Local) variable); // 对parameter variable进行故障注入
        } else if (scope.equals("field")) {
            return this.injectFieldWithNull(b, (SootField) variable); // 对class的field进行故障注入
        } else if (scope.equals("return")) {
            return this.injectReturnWithNull(b, (Stmt) variable); // 对return variable进行故障注入
        }
        return false;
    }

    // 对Field variable进行故障注入
    private boolean injectFieldWithNull(Body b, SootField field) {
        this.injectInfo.put("VariableName", field.getName());
        Chain<Unit> units = b.getUnits();// 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator();// 获取代码语句的快照迭代器
        while (unitIt.hasNext()) {
            Unit tmp = unitIt.next();
            Iterator<ValueBox> valueBoxes = tmp.getUseAndDefBoxes().iterator();
            while (valueBoxes.hasNext()) {
                Value value = valueBoxes.next().getValue();
                // check whether SootField in Soot is in SingleTon mode for equals comparison
                if ((value instanceof FieldRef) && (((FieldRef) value).getField().equals(field))) {// 定位到目标Field，此时tmp指向目标Field
                    logger.debug(tmp.toString());
                    try {
                        List<Stmt> stmts = this.getNullFieldStatements(b, field);// 获取Value故障注入的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                // here we add the field null to the begining of the method
                                units.insertBefore(stmts.get(i), units.getFirst());// 首句在tmp前插入
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
                        List<Stmt> conditionStmts = this.getConditionStmt(b, tmp);// 获取条件语句
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
                    } catch (Exception e) {
                        logger.error(e.getMessage());
                        logger.error(this.formatInjectionInfo());
                        return false;
                    }

                }
            }

        }
        return false;
    }

    // 获取针对Field variable的故障注入的语句
    private List<Stmt> getNullFieldStatements(Body b, SootField field) {
        Stmt nullFieldStmt = null;
        boolean isStaticField = false;
        if (((field.getModifiers() & Modifier.STATIC) ^ Modifier.STATIC) == 0) {//判断该字段是否为静态字段
            isStaticField = true;// 修改字段静态标识
            this.injectInfo.put("static", "true"); // 记录该字段为静态字段
        }

        if (isStaticField) {// 若该字段为静态字段
            nullFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), NullConstant.v()); // 赋值语句 static field = null
        } else {
            nullFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
                    NullConstant.v()); // 赋值语句 field = null
        }

        List<Stmt> stmts = new ArrayList<Stmt>();
        stmts.add(nullFieldStmt); // 记录赋值语句 field = null
        return stmts; // 返回故障注入语句
    }

    // 对Local variable进行故障注入
    private boolean injectLocalWithNull(Body b, Local local) {
        this.injectInfo.put("VariableName", local.getName());
        logger.debug(this.formatInjectionInfo());
        Chain<Unit> units = b.getUnits(); // 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator(); // 获取代码语句的快照迭代器
        while (unitIt.hasNext()) {
            Unit tmp = unitIt.next();
            Iterator<ValueBox> valueBoxes = tmp.getUseBoxes().iterator();
            while (valueBoxes.hasNext()) {
                Value value = valueBoxes.next().getValue();
                if ((value instanceof Local) && (value.equivTo(local))) { // 定位到目标Local variable
                    logger.debug(tmp.toString());
                    try {
                        List<Stmt> stmts = this.getNullLocalStatements(local); // 获取故障注入后的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                // here is the difference between local and parameter variable
                                units.insertBefore(stmts.get(i), tmp); // 首句在tmp前插入
                            } else {
                                units.insertAfter(stmts.get(i), stmts.get(i - 1)); // 之后每句在前一句之后插入
                            }
                        }
                        List<Stmt> preStmts = this.getPrecheckingStmts(b); // 获取检查语句，检测激活模式，
                        for (int i = 0; i < preStmts.size(); i++) {
                            if (i == 0) {
                                units.insertBefore(preStmts.get(i), stmts.get(0)); // 首句在stmts[0]前插入
                            } else {
                                units.insertAfter(preStmts.get(i), preStmts.get(i - 1));// 之后每句在前一句之后插入
                            }
                        }
                        List<Stmt> conditionStmts = this.getConditionStmt(b, tmp);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
            }

        }
        return false;
    }

    // 对Return variable进行故障注入
    private boolean injectReturnWithNull(Body b, Stmt stmt) {
        Chain<Unit> units = b.getUnits(); // 获取当前代码语句
        try {
            Stmt targetStmt = stmt;
            if (targetStmt instanceof ReturnStmt) { // 定位到return语句
                Stmt nullReturnStmt = this.getNullReturnStatements();// 获取注入后的代码语句
                units.insertBefore(nullReturnStmt, targetStmt);// 首句在return语句前插入
                List<Stmt> preStmts = this.getPrecheckingStmts(b); // 获取检查语句，检测激活模式，
                for (int i = 0; i < preStmts.size(); i++) {
                    if (i == 0) {
                        units.insertBefore(preStmts.get(i), nullReturnStmt);// 首句在stmts[0]前插入
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

            }
        } catch (Exception e) {
            logger.error(e.getMessage());
            logger.error(this.formatInjectionInfo());
            return false;
        }

        return false;
    }

    // 获取针对Return variable的故障注入的语句
    private Stmt getNullReturnStatements() {
        ReturnStmt returnStmt = Jimple.v().newReturnStmt(NullConstant.v()); // 返回语句 return null
        return returnStmt; // 返回故障注入语句
    }

    // 对Parameter variable进行故障注入
    private boolean injectParameterWithNull(Body b, Local local) {
        this.injectInfo.put("VariableName", local.getName());
        logger.debug(this.formatInjectionInfo());
        Chain<Unit> units = b.getUnits();// 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator();// 获取代码语句的快照迭代器
        while (unitIt.hasNext()) {
            Unit tmp = unitIt.next();
            Iterator<ValueBox> valueBoxes = tmp.getUseBoxes().iterator();
            while (valueBoxes.hasNext()) {
                Value value = valueBoxes.next().getValue();
                if ((value instanceof Local) && (value.equivTo(local))) {// 定位到目标parameter variable
                    logger.debug(tmp.toString());
                    try {

                        List<Stmt> stmts = this.getNullLocalStatements(local);// 获取故障注入后的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                // before useing the local parameter
                                units.insertBefore(stmts.get(i), tmp);// 首句在tmp前插入
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
                        List<Stmt> conditionStmts = this.getConditionStmt(b, tmp);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
            }
        }
        return false;
    }

    // 获取针对Local variable的故障注入的语句
    private List<Stmt> getNullLocalStatements(Local local) {
        List<Stmt> stmts = new ArrayList<Stmt>();
        Stmt nullLocalStmt = null;
        nullLocalStmt = Jimple.v().newAssignStmt(local, NullConstant.v()); // 赋值语句 local = null
        stmts.add(nullLocalStmt); // 记录赋值语句 local = null
        return stmts; // 返回故障注入语句
    }

    // 将body内的variable按照域分组返回
    private Map<String, List<Object>> getAllVariables(Body b) {
        Map<String, List<Object>> result = new HashMap<String, List<Object>>();// 声明结果Map
        List<Local> tmpPLocals = b.getParameterLocals();// 获取全部的Parameter变量
        List<Object> pLocals = new ArrayList<Object>();// 声明符合要求的，参数类型的variable集合
        Chain<Local> tmpLocals = b.getLocals();// 获取全部的Local变量
        List<Object> locals = new ArrayList<Object>();// 声明符合要求的，局部变量类型的variable集合
        Chain<SootField> tmpFields = b.getMethod().getDeclaringClass().getFields();// 通过函数声明找到类声明再找到所有的Field变量
        List<Object> fields = new ArrayList<Object>();// 声明符合要求的，字段类型的variable集合

        List<Object> returnStmt = new ArrayList<Object>();

        int i;
        for (i = 0; i < tmpPLocals.size(); i++) {// 遍历每一个Parameter
            Local local = tmpPLocals.get(i);// 获取该Parameter
            if (this.isTargetedType(local.getType()) && (!local.getName().startsWith("$"))) {// 若该参数的类型在处理范围之内
                pLocals.add(local);// 在参数类型的variable集合中，添加该variable
            }
        }

        Iterator<Local> tmpLocalsItr = tmpLocals.iterator();// 获取Local变量迭代器
        while (tmpLocalsItr.hasNext()) {// 遍历每一个Local
            Local local = tmpLocalsItr.next();// 获取该Local
            if (!pLocals.contains(local)) {// 如果该Local不是parameter
                if (this.isTargetedType(local.getType()) && (!local.getName().startsWith("$"))
                        && (!local.getName().equals("this"))) {// 若该参数的类型在处理范围之内
                    locals.add(local);// 在局部变量类型的variable集合中，添加该variable
                }
            }

        }

        // actually soot cannot get Class Fields
        Iterator<SootField> tmpFieldItr = tmpFields.iterator();// 获取Field变量迭代器
        while (tmpFieldItr.hasNext()) {// 遍历每一个Field
            SootField field = tmpFieldItr.next();// 获取该Field
            if (this.isTargetedType(field.getType()) && (!field.getName().startsWith("$"))) {// 若该参数的类型在处理范围之内
                if (((field.getModifiers() & Modifier.FINAL) ^ Modifier.FINAL) == 0) {// not sure
                    // this is a final field, we cannot change value of it
//					logger.info(field.getDeclaration());
                } else {
                    fields.add(field);// 在字段类型的variable集合中，添加该variable
                }
            }
        }

        // for the variable used by return
        Type returnType = b.getMethod().getReturnType();// 获取返回类型
//		if (!(returnType instanceof VoidType)) {
        if (this.isTargetedType(returnType)) { // 若返回类型为目标类型
            Iterator<Unit> unitItr = b.getUnits().snapshotIterator();// 获取代码语句的快照迭代器
            while (unitItr.hasNext()) {
                Unit unit = unitItr.next();
                if (unit instanceof ReturnStmt) {// 定位到return 语句
                    returnStmt.add(unit); // 记录返回语句
                }
            }
        }

        result.put("parameter", pLocals);// 在结果Map中添加符合要求的parameter类variable
        result.put("local", locals);// 在结果Map中添加符合要求的local类variable
        result.put("field", fields);// 在结果Map中添加符合要求的field类variable
        result.put("return", returnStmt); // 在结果Map中添加符合要求的return语句
        return result;// 返回结果Map
    }

    // 判断是否为目标类型
    private boolean isTargetedType(Type type) {
        // only non-primitive types are included
        if (type instanceof BooleanType) {
            return false;
        } else if (type instanceof ByteType) {
            return false;
        } else if (type instanceof ShortType) {
            return false;
        } else if (type instanceof IntType) {
            return false;
        } else if (type instanceof LongType) {
            return false;
        } else if (type instanceof FloatType) {
            return false;
        } else if (type instanceof DoubleType) {
            return false;
        } else return !(type instanceof VoidType);
    }

    // 获取全部符合待注入要求的variable
    private List<Object> getQualifiedVariables(Body b, Map<String, List<Object>> allVariables, String scope) {
        // just copy list
        List<Object> variablesInScope = allVariables.get(scope);
        return new ArrayList<Object>(variablesInScope); // 返回目标域内的所有variable
    }

    // we should decide whether choose return as a scope
    private List<String> getTargetScope(String variableScope) {
        List<String> scopes = new ArrayList<String>();// 声明待注入域集合列表
        if ((variableScope == null) || (variableScope == "")) {// 若未指定待注入域
            scopes.add("local");// 待注入域集合添加"local"类
            scopes.add("field");// 待注入域集合添加"field"类
            scopes.add("parameter");// 待注入域集合添加"parameter"类
            scopes.add("return");// 待注入域集合添加"return"类
        } else {// 若指定了待注入域
            scopes.add(variableScope);// 直接在注入域集合添加指定的域类型
        }
        return scopes;
    }
}
