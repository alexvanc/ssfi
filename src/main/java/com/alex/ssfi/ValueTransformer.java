package com.alex.ssfi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.RunningParameter;

import soot.Body;
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
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.AddExpr;
import soot.jimple.DoubleConstant;
import soot.jimple.FieldRef;
import soot.jimple.FloatConstant;
import soot.jimple.IntConstant;
import soot.jimple.Jimple;
import soot.jimple.LongConstant;
import soot.jimple.Stmt;
import soot.jimple.StringConstant;
import soot.jimple.SubExpr;
import soot.util.Chain;

public class ValueTransformer extends BasicTransformer {

    private final Logger logger = LogManager.getLogger(ValueTransformer.class);

    public ValueTransformer(RunningParameter parameters) {
        super(parameters);
    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) { // 核心的回调方法
        while (!this.parameters.isInjected()) {
            // in this way, all the FIs are performed in the first function of this class
            SootMethod targetMethod = this.generateTargetMethod(b); // 获取待注入的Method（随机或指定）
            if (targetMethod == null) { // 若获取不到待注入的Method，则返回上级程序
                return;
            }
            Body tmpBody;
            try {
                tmpBody = targetMethod.retrieveActiveBody(); // 获取Method对应的Body
            } catch (Exception e) { // 若获取Method对应的Body失败，则跳过本次Method，并尝试获取下一个待注入的Method
                logger.info("Retrieve Active Body Failed!");
                continue;
            }
            if (tmpBody == null) { // 如果获取的Body为空，则跳过本次Method，并尝试获取下一个待注入的Method
                continue;
            }
            this.startToInject(tmpBody); // 开始故障注入。对待注入的Method对应的Body进行故障注入
        }

    }

    // 对于目标Body进行故障注入
    private void startToInject(Body b) {
        // no matter this inject fails or succeeds, this targetMethod is already used
        SootMethod targetMethod = b.getMethod(); // 获取目标Method
        this.injectInfo.put("FaultType", "Value_FAULT"); // 标记故障类型为，"Value_FAULT"
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName()); // 获取目标包名
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName()); // 获取目标类名
        this.injectInfo.put("Method", targetMethod.getSubSignature()); // 获取目标函数签名
        List<String> scopes = getTargetScope(this.parameters.getVariableScope()); // 获取目标域集合

        Map<String, List<Object>> allVariables = this.getAllVariables(b); // 获取目标Body相关的所有variable，并按照Field、Local、Parameter分组保存
        logger.debug("Try to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // randomly combine scope, type , variable, action
        while (true) {
            int scopesSize = scopes.size(); // 获取待注入的域集合
            if (scopesSize == 0) {
                logger.debug("Cannot find qualified scopes");
                break;
            }
            int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize); // 随机选择待注入的域的索引
            String scope = scopes.get(scopeIndex); // 获取待注入的域
            scopes.remove(scopeIndex); // 在待注入的域集合中，移除将要进行尝试的域
            this.injectInfo.put("VariableScope", scope); // 记录待注入的域
            List<String> types = getTargetType(this.parameters.getVariableType()); // 获取目标类型集合
            // randomly combine type, variable, action
            while (true) {
                int typesSize = types.size(); // 获取待注入variable的类型集合
                if (typesSize == 0) {
                    break;
                }
                int typeIndex = new Random(System.currentTimeMillis()).nextInt(typesSize); // 随机选择待注入variable的类型的索引
                String type = types.get(typeIndex); // 获取待注入variable的类型
                types.remove(typeIndex); // 在待注入variable的类型集合中，移除将要进行尝试的类型
                this.injectInfo.put("VariableType", type); // 记录待注入的类型
                // randomly combine variable, action
                List<Object> qualifiedVariables = this.getQualifiedVariables(b, allVariables, scope, type); // 获取目标域目标类型的variable集合，作为待注入的variable集合
                while (true) {
                    int variablesSize = qualifiedVariables.size(); // 获取待注入variable集合的大小
                    if (variablesSize == 0) {
                        break;
                    }
                    int variableIndex = new Random(System.currentTimeMillis()).nextInt(variablesSize); // 随机选择待注入variable的索引
                    Object variable = qualifiedVariables.get(variableIndex); // 获取待注入的variable
                    qualifiedVariables.remove(variableIndex); // 在待注入的variable集合中，移除将要进行尝试的variable
                    // randomly generate an action for this variable
                    List<String> possibleActions = this.getPossibleActionsForVariable(type); // 获取所有可行的操作行为，作为待注入的操作集合
                    while (true) {
                        int actionSize = possibleActions.size(); // 获取待注入的操作集合的大小
                        if (actionSize == 0) {
                            break;
                        }
                        int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize); // 随机选择待注入的操作的索引
                        String action = possibleActions.get(actionIndex); // 获取待注入操作
                        possibleActions.remove(actionIndex); // 在待注入的操作的集合中，移除将要进行尝试的操作
                        this.injectInfo.put("Action", action);
                        // finally perform injection
                        if (this.inject(b, scope, variable, action)) { // 进行注入
                            this.parameters.setInjected(true); // 标记注入成功
                            this.recordInjectionInfo(); // 记录注入信息
                            logger.debug("Succeed to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
                                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
                            return;
                        } else {
                            logger.debug("Failed injection:+ " + this.formatInjectionInfo());
                        }
                    }
                }
            }
            logger.debug("Fail to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
        }
    }

    // 执行注入操作
    private boolean inject(Body b, String scope, Object variable, String action) {
        if (scope.equals("field")) {// inject fault to field variables
            SootField field = (SootField) variable; // 将待注入的variable转换为Soot字段类型
            this.injectInfo.put("VariableName", field.getName()); // 记录注入的variable名
            this.injectInfo.put("Action", action); // 记录注入的操作类型
            return this.injectFieldWithAction(b, field, action); // 对class的field进行故障注入
        } else if (scope.equals("local")) {// inject this fault to local or parameter variables
            Local local = (Local) variable; // 将待注入的variable转换为Soot局部变量类型
            this.injectInfo.put("VariableName", local.getName());
            this.injectInfo.put("Action", action);
            return this.injectLocalWithAction(b, local, action); // 对local variable 进行故障注入
        } else if (scope.equals("parameter")) { // 若待注入的域是Method的参数域
            Local local = (Local) variable; // 将待注入的variable转换为Soot参数类型
            this.injectInfo.put("VariableName", local.getName());
            this.injectInfo.put("Action", action);
            return this.injectParameterWithAction(b, local, action); // 对parameter variable进行故障注入
        }
        return false;
    }

    // 对于给定的待注入variable的类型，获取所有可能的注入操作类型
    private List<String> getPossibleActionsForVariable(String type) {
        // For simplicity, we don't check whether the action is applicable to this type
        // of variables
        // this should be done in the configuration validator
        List<String> possibleActions = new ArrayList<String>();  // 声明可能的注入操作类型集合的列表
        String specifiedAction = this.parameters.getAction(); // 记录配置指定的操作类型
        String specifiedValue = this.parameters.getVariableValue(); // 记录配置指定的目标value，而非待注入variable
        if ((specifiedAction == null) || (specifiedAction == "")) { //若指定的操作类型为空，
            if (type.equals("java.lang.String")) { // 若待注入variable的类型为String
                possibleActions.add("TO"); // 可能的注入操作为"TO"，即将目标value赋值给待注入variable
            } else { // 若待注入variable为其他类型
                if ((specifiedValue != null) && (specifiedValue != "")) { // 若指定的目标value不为空
                    possibleActions.add("ADD"); // 可能的注入操作为"ADD"，即将待注入把variable与目标value相加的结果再赋值给待注入variable
                    possibleActions.add("SUB"); // 可能的注入操作为"SUB"，即将待注入把variable与目标value相减的结果再赋值给待注入variable
                    possibleActions.add("TO"); // 可能的注入操作为"TO"
                } else { // 若指定的目标value为空
                    possibleActions.add("ADD"); // 可能的注入操作为"ADD"
                    possibleActions.add("SUB"); // 可能的注入操作为"SUB"
                }
            }
        } else { // 若指定了操作类型
            possibleActions.add(specifiedAction);  // 可能的注入操作为指定的操作
        }
        return possibleActions; // 返回可能的操作集合
    }

    // 获取全部符合待注入要求的variable
    private List<Object> getQualifiedVariables(Body b, Map<String, List<Object>> allVariables, String scope,
                                               String type) {
        List<Object> qualifiedVariables = new ArrayList<Object>(); // 声明符合要求的variable列表
        List<Object> variablesInScope = allVariables.get(scope); // 获取指定scope内的全部variable
        int length = variablesInScope.size(); // 获取指定scope内的variable的数量
        String targetVariableName = this.parameters.getVariableName(); // 获取目标variable的名字
        if (scope.equals("local") || scope.equals("parameter")) { // 若指定scope为"local"或"parameter"
            for (int i = 0; i < length; i++) { // 遍历每一个variable
                Local localVariable = (Local) variablesInScope.get(i); // 将当前variable转换为Local类型
                if (localVariable.getType().toString().equals(type)) { // 校验variable的类型是否为目标类型
                    // if users specify the variableName
                    if ((targetVariableName == null) || (targetVariableName == "")) { // 若目标variable的名字为空，则将所有目标类型的variable添加到qualifiedVariables
                        qualifiedVariables.add(localVariable);
                    } else {
                        if (localVariable.getName().equals(targetVariableName)) { //若目标variable的名字非空，且当前variable的名字匹配到目标名字，则将当前variable添加到qualifiedVariables
                            qualifiedVariables.add(localVariable);
                        }
                    }
                }
            }
        } else {// scope is field
            for (int i = 0; i < length; i++) {
                SootField fieldVariable = (SootField) variablesInScope.get(i); // 将当前variable转换为SootField类型
                if (fieldVariable.getType().toString().equals(type)) {
                    // if users specify the variableName
                    if ((targetVariableName == null) || (targetVariableName == "")) {  // 若目标variable的名字为空，则将所有目标类型的variable添加到qualifiedVariables
                        qualifiedVariables.add(fieldVariable);
                    } else {
                        if (fieldVariable.getName().equals(targetVariableName)) { //若目标variable的名字非空，且当前variable的名字匹配到目标名字，则将当前variable添加到qualifiedVariables
                            qualifiedVariables.add(fieldVariable);
                        }
                    }
                }
            }
        }
        return qualifiedVariables; //返回符合要求的variable列表
    }

    // 获取待注入的Method
    private synchronized SootMethod generateTargetMethod(Body b) {
        if (this.allQualifiedMethods == null) { // 若符合要求的Method集合为null
            this.initAllQualifiedMethods(b); // 初始化符合要求的Method集合
        }
        int leftQualifiedMethodsSize = this.allQualifiedMethods.size(); // 获取符合要求的Method集合的大小
        if (leftQualifiedMethodsSize == 0) { // 若符合要求的Method集合的大小为0，则返回null
            return null;
        }
        int randomMethodIndex = new Random(System.currentTimeMillis()).nextInt(leftQualifiedMethodsSize); // 随机选择待注入Method的索引

        SootMethod targetMethod = this.allQualifiedMethods.get(randomMethodIndex); // 获取待注入的Method
        this.allQualifiedMethods.remove(randomMethodIndex); // 在待注入Method的集合中，移除将要进行尝试的Method
        return targetMethod; // 返回待注入的Method
    }

    // for this fault type,we simply assume all methods satisfy the condition
    private void initAllQualifiedMethods(Body b) {
        List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods(); // 获取当前body能够获取的全部Method
        List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>(); // 声明符合要求的Method列表
        boolean withSpefcifiedMethod = true;
        String specifiedMethodName = this.parameters.getMethodName(); // 获取指定的目标Method名字
        if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) { // 如果未指定待注入的Method
            withSpefcifiedMethod = false; // 记录标记
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

    // 给定注入的操作类型，对Local variable进行故障注入
    private boolean injectLocalWithAction(Body b, Local local, String action) {
        // inject at the beginning or inject after assignment make a lot of difference
        // currently we inject the fault after it is visited once
        logger.info(this.formatInjectionInfo());
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
                        List<Stmt> stmts = this.getLocalStatementsByAction(local, action); // 获取注入后的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                // here is the difference between local and parameter variable
                                units.insertBefore(stmts.get(i), tmp); // 首句在tmp前插入
                            } else {
                                units.insertAfter(stmts.get(i), stmts.get(i - 1));// 之后每句在前一句之后插入
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
                        List<Stmt> conditionStmts = this.getConditionStmt(b, tmp); // 获取条件语句，根据激活模式或条件判断是否激活故障
                        for (int i = 0; i < conditionStmts.size(); i++) {
                            if (i == 0) {
                                units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1)); // 首句在preStmts[-1]后插入
                            } else {
                                units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));// 之后每句在前一句之后插入
                            }
                        }
                        List<Stmt> actStmts = this.createActivateStatement(b); // 获取故障注入后的操作语句，记录日志等
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

    // 给定注入的操作类型，对parameter variable进行故障注入
    private boolean injectParameterWithAction(Body b, Local local, String action) {
        // inject at the beginning or inject after assignment make a lot of difference
        // currently we inject the fault after it is visited once
        logger.info(this.formatInjectionInfo());
        Chain<Unit> units = b.getUnits();// 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator();// 获取代码语句的快照迭代器
        while (unitIt.hasNext()) {
            Unit tmp = unitIt.next();
            Iterator<ValueBox> valueBoxes = tmp.getUseAndDefBoxes().iterator();
            while (valueBoxes.hasNext()) {
                Value value = valueBoxes.next().getValue();
                if ((value instanceof Local) && (value.equivTo(local))) {// 定位到目标parameter variable
                    logger.debug(tmp.toString());
                    Unit nextUnit = units.getSuccOf(tmp);
                    try {
                        List<Stmt> stmts = this.getLocalStatementsByAction(local, action); // 获取故障注入后的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                units.insertAfter(stmts.get(i), tmp);// 首句在tmp后插入
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
                        List<Stmt> conditionStmts = this.getConditionStmt(b, nextUnit);// 获取条件语句，根据激活模式或条件判断是否激活故障
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
    private List<Stmt> getLocalStatementsByAction(Local local, String action) {
        Stmt valueChangeStmt = null; // 声明值改变语句对象
        if (action.equals("TO")) { // 若故障注入操作为"TO"
            String typeName = local.getType().toString();  // 获取待注入variable的类型名称
            String targetStringValue = this.parameters.getVariableValue(); // 获取目标value，非待注入variable
            if (targetStringValue != null) { // 如果目标value非空
                this.injectInfo.put("VariableValue", targetStringValue); // 记录目标value
            }
            if (typeName.equals("byte")) { // 若待注入variable的类型是"byte"
                byte result = Byte.parseByte(targetStringValue); // 获取目标byte value
                valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result)); // 获取local变量的byte value改变语句
            } else if (typeName.equals("short")) { // 若待注入variable的类型是"short"
                short result = Short.parseShort(targetStringValue);// 获取目标short value
                valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result));// 获取local变量的short value改变语句
            } else if (typeName.equals("int")) {// 若待注入variable的类型是"int"
                int result = Integer.parseInt(targetStringValue);// 获取目标int value
                valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result));// 获取local变量的int value改变语句
            } else if (typeName.equals("long")) {// 若待注入variable的类型是"long"
                long result = Long.parseLong(targetStringValue);// 获取目标long value
                valueChangeStmt = Jimple.v().newAssignStmt(local, LongConstant.v(result));// 获取local变量的long value改变语句
            } else if (typeName.equals("float")) {// 若待注入variable的类型是"float"
                float result = Float.parseFloat(targetStringValue);// 获取目标float value
                valueChangeStmt = Jimple.v().newAssignStmt(local, FloatConstant.v(result));// 获取local变量的float value改变语句
            } else if (typeName.equals("double")) {// 若待注入variable的类型是"double"
                double result = Double.parseDouble(targetStringValue);// 获取目标double value
                valueChangeStmt = Jimple.v().newAssignStmt(local, DoubleConstant.v(result));// 获取local变量的double value改变语句
            } else if (typeName.equals("java.lang.String")) {// 若待注入variable的类型是"string"
                if ((targetStringValue == null) || (targetStringValue == "")) {// 如果目标value为空
                    targetStringValue = "test";// 初始化目标string value
                }
                valueChangeStmt = Jimple.v().newAssignStmt(local, StringConstant.v(targetStringValue));// 获取local变量的string value改变语句
            }
        } else if (action.equals("ADD")) {// 若待注入variable的类型是"ADD"
            String typeName = local.getType().toString(); // 获取待注入variable的类型名称
            String addedStringValue = this.parameters.getVariableValue(); // 获取目标value，非待注入variable
            if (addedStringValue != null) { // 如果目标value非空
                this.injectInfo.put("VariableValue", addedStringValue);// 记录目标value
            }
            if (typeName.equals("byte")) {// 若待注入variable的类型是"byte"
                byte addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Byte.parseByte(addedStringValue);// 获取目标byte value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue)); // 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp); // 创建赋值表达式对象；获取byte value的改变语句
            } else if (typeName.equals("short")) {// 若待注入variable的类型是"short"
                short addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Short.parseShort(addedStringValue);// 获取目标short value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取short value的改变语句
            } else if (typeName.equals("int")) {// 若待注入variable的类型是"int"
                int addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Integer.parseInt(addedStringValue);// 获取目标int value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取int value的改变语句
            } else if (typeName.equals("long")) {// 若待注入variable的类型是"long"
                long addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Long.parseLong(addedStringValue);// 获取目标long value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, LongConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取long value的改变语句
            } else if (typeName.equals("float")) {// 若待注入variable的类型是"float"
                float addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Float.parseFloat(addedStringValue);// 获取目标float value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取float value的改变语句
            } else if (typeName.equals("double")) {// 若待注入variable的类型是"double"
                double addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {// 如果目标value非空
                    addedValue = Double.parseDouble(addedStringValue);// 获取目标double value
                }
                AddExpr addExp = Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取float value的改变语句
            }
        } else if (action.equals("SUB")) {// 若待注入variable的类型是"ADD"
            String typeName = local.getType().toString(); // 获取待注入variable的类型名称
            String subedStringValue = this.parameters.getVariableValue(); // 获取目标value，非待注入variable
            if (subedStringValue != null) { // 如果目标value非空
                this.injectInfo.put("VariableValue", subedStringValue);// 记录目标value
            }
            if (typeName.equals("byte")) {// 若待注入variable的类型是"byte"
                byte addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {// 如果目标value非空
                    addedValue = Byte.parseByte(subedStringValue);// 获取目标byte value
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句
            } else if (typeName.equals("short")) {// 若待注入variable的类型是"byte"
                short addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {
                    addedValue = Short.parseShort(subedStringValue);
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
            } else if (typeName.equals("int")) {// 若待注入variable的类型是"int"
                int addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {// 如果目标value非空
                    addedValue = Integer.parseInt(subedStringValue);// 获取目标int value
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句

            } else if (typeName.equals("long")) {// 若待注入variable的类型是"long"
                long addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {// 如果目标value非空
                    addedValue = Long.parseLong(subedStringValue);// 获取目标long value
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, LongConstant.v(addedValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句

            } else if (typeName.equals("float")) {// 若待注入variable的类型是"float"
                float addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {// 如果目标value非空
                    addedValue = Float.parseFloat(subedStringValue);// 获取目标float value
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, FloatConstant.v(addedValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句
            } else if (typeName.equals("double")) {// 若待注入variable的类型是"double"
                double addedValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {// 如果目标value非空
                    addedValue = Double.parseDouble(subedStringValue);// 获取目标double value
                }
                SubExpr addExp = Jimple.v().newSubExpr(local, DoubleConstant.v(addedValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句
            }
        }
        List<Stmt> stmts = new ArrayList<Stmt>();
        stmts.add(valueChangeStmt);// 记录value改变语句 // local = targetStringValue // local = local + targetStringValue // local = local - targetStringValue
        return stmts; // 返回待插入的语句
    }

    // 获取针对Field variable的故障注入的语句
    private List<Stmt> getFieldStatementsByAction(Body b, SootField field, String action) {
        Stmt valueChangeStmt = null; // 声明值改变语句
        Stmt copyFieldStmt = null; // 声明字段考培语句
        Stmt changeFieldStmt = null; // 声明字段改变语句
        boolean isStaticField = false; // 声明字段静态标识
        if (((field.getModifiers() & Modifier.STATIC) ^ Modifier.STATIC) == 0) { //判断该字段是否为静态字段
            isStaticField = true; // 修改字段静态标识
            this.injectInfo.put("static", "true"); // 记录该字段为静态字段
        }

        List<Stmt> stmts = new ArrayList<Stmt>(); // 声明语句集合列表
        Local local = null; // 临时local变量
        if (action.equals("TO")) { // 若故障注入操作为"TO"
            String typeName = field.getType().toString(); // 获取待注入variable的类型名称
            String targetStringValue = this.parameters.getVariableValue(); // 获取目标value，非待注入variable
            if (targetStringValue != null) { // 如果目标value非空
                this.injectInfo.put("VariableValue", targetStringValue); // 记录目标value
            }
            if (typeName.equals("byte")) { // 若待注入variable的类型是"byte"
                byte result = Byte.parseByte(targetStringValue); // 获取目标byte value
                if (isStaticField) { // 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            IntConstant.v(result)); // 获取静态字段的byte value改变语句
                } else { // 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result)); // 获取非静态字段的byte value改变语句
                }
            } else if (typeName.equals("short")) { // 若待注入variable的类型是"short"
                short result = Short.parseShort(targetStringValue); // 获取目标short value
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            IntConstant.v(result));// 获取静态字段的short value改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result));// 获取非静态字段的short value改变语句
                }
            } else if (typeName.equals("int")) {// 若待注入variable的类型是"int"
                int result = Integer.parseInt(targetStringValue);// 获取目标int value
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            IntConstant.v(result));// 获取静态字段的int value改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result));// 获取非静态字段的int value改变语句
                }
            } else if (typeName.equals("long")) {// 若待注入variable的类型是"long"
                long result = Long.parseLong(targetStringValue); // 获取目标long value
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            LongConstant.v(result));// 获取静态字段的long value改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), LongConstant.v(result)); // 获取非静态字段的long value改变语句
                }
            } else if (typeName.equals("float")) {// 若待注入variable的类型是"float"
                float result = Float.parseFloat(targetStringValue);// 获取目标float value
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            FloatConstant.v(result));// 获取静态字段的float value改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), FloatConstant.v(result));// 获取非静态字段的float value改变语句
                }
            } else if (typeName.equals("double")) {// 若待注入variable的类型是"double"
                double result = Double.parseDouble(targetStringValue);// 获取目标double value
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            DoubleConstant.v(result));// 获取静态字段的double value改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
                            DoubleConstant.v(result));// 获取非静态字段的double value改变语句
                }
            } else if (typeName.equals("java.lang.String")) {// 若待注入variable的类型是"java.lang.String"
                if ((targetStringValue == null) || (targetStringValue == "")) {// 如果未指定目标value
                    targetStringValue = "test";// 设定目标value为"test"
                }
                if (isStaticField) {// 若该字段为静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
                            StringConstant.v(targetStringValue));// 获取静态字段的string value的改变语句
                } else {// 若该字段为非静态字段
                    valueChangeStmt = Jimple.v().newAssignStmt(
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
                            StringConstant.v(targetStringValue));// 获取非静态字段的string value的改变语句
                }
            }
            stmts.add(valueChangeStmt);// 记录value改变语句 // field = targetStringValue
        } else if (action.equals("ADD")) {// 若故障注入操作为"ADD"
            String typeName = field.getType().toString();// 获得待注入variable的类型名称
            String addedStringValue = this.parameters.getVariableValue();// 获取目标value
            if (addedStringValue != null) {// 如果目标value非空
                this.injectInfo.put("VariableValue", addedStringValue);// 记录目标value
            }
            if (typeName.equals("byte")) {// 若待注入variable的类型为"byte"
                byte addedValue = 1;// 初始化
                if ((addedStringValue != null) && (addedStringValue != "")) {//目标vale不为空
                    addedValue = Byte.parseByte(addedStringValue);// 获取目标byte value
                }
                local = Jimple.v().newLocal("field_tmp", ByteType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("short")) { // 若待注入variable的类型为"short"
                short addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) { //目标vale不为空
                    addedValue = Short.parseShort(addedStringValue);// 获取目标short value
                }
                local = Jimple.v().newLocal("field_tmp", ShortType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取short value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("int")) {// 若待注入variable的类型为"int"
                int addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {  //若目标vale不为空
                    addedValue = Integer.parseInt(addedStringValue); // 获取目标int value
                }
                local = Jimple.v().newLocal("field_tmp", IntType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取int value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("long")) {// 若待注入variable的类型为"long"
                long addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {//若目标vale不为空
                    addedValue = Long.parseLong(addedStringValue);// 获取目标long value
                }
                local = Jimple.v().newLocal("field_tmp", LongType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, LongConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取long value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("float")) {// 若待注入variable的类型为"float"
                float addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {//若目标vale不为空
                    addedValue = Float.parseFloat(addedStringValue);// 获取目标float value
                }
                local = Jimple.v().newLocal("field_tmp", FloatType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取float value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("double")) {// 若待注入variable的类型为"double"
                double addedValue = 1;
                if ((addedStringValue != null) && (addedStringValue != "")) {//若目标vale不为空
                    addedValue = Double.parseDouble(addedStringValue);// 获取目标double value
                }
                local = Jimple.v().newLocal("field_tmp", DoubleType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                AddExpr addExp = Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));// 创建加法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取double value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            }
            stmts.add(copyFieldStmt);// 记录复制字段语句 // field_tmp = field
            stmts.add(valueChangeStmt);// 记录value改变语句 // field_tmp = field_tmp + addedValue
            stmts.add(changeFieldStmt);// 记录改变字段语句 // field = field_tmp
        } else if (action.equals("SUB")) { // 若故障注入操作为"SUB"
            String typeName = field.getType().toString(); // 获得待注入variable的类型名称
            String subedStringValue = this.parameters.getVariableValue(); // 获取目标value，非待注入variable
            if (subedStringValue != null) { // 如果目标value非空
                this.injectInfo.put("VariableValue", subedStringValue); // 记录目标value
            }
            if (typeName.equals("byte")) {// 若待注入variable的类型为"byte"
                byte subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Byte.parseByte(subedStringValue);// 获取目标byte value
                }
                local = Jimple.v().newLocal("field_tmp", ByteType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取byte value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("short")) {// 若待注入variable的类型为"short"
                short subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Short.parseShort(subedStringValue);// 获取目标short value
                }
                local = Jimple.v().newLocal("field_tmp", ShortType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取short value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("int")) {// 若待注入variable的类型为"int"
                int subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Integer.parseInt(subedStringValue);// 获取目标int value
                }
                local = Jimple.v().newLocal("field_tmp", IntType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取short value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("long")) {// 若待注入variable的类型为"long"
                long subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Long.parseLong(subedStringValue);// 获取目标long value
                }
                local = Jimple.v().newLocal("field_tmp", LongType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, LongConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取long value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("float")) {// 若待注入variable的类型为"float"
                float subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Float.parseFloat(subedStringValue);// 获取目标float value
                }
                local = Jimple.v().newLocal("field_tmp", FloatType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, FloatConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取float value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            } else if (typeName.equals("double")) {// 若待注入variable的类型为"double"
                double subededValue = 1;
                if ((subedStringValue != null) && (subedStringValue != "")) {//若目标vale不为空
                    subededValue = Double.parseDouble(subedStringValue);// 获取目标double value
                }
                local = Jimple.v().newLocal("field_tmp", DoubleType.v());// 创建局部临时变量对象
                b.getLocals().add(local);// 在局部变量域，添加该局部临时变量
                SubExpr addExp = Jimple.v().newSubExpr(local, DoubleConstant.v(subededValue));// 创建减法表达式对象
                valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);// 创建赋值表达式对象；获取double value的改变语句
                if (isStaticField) {
                    copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));// 创建复制静态字段的语句
                    changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);// 创建改变静态字段的语句
                } else {
                    copyFieldStmt = Jimple.v().newAssignStmt(local,
                            Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));// 创建复制非静态字段的语句
                    changeFieldStmt = Jimple.v()
                            .newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);// 创建改变非静态字段的语句
                }
            }
            stmts.add(copyFieldStmt); // 记录复制字段语句 // field_tmp = field
            stmts.add(valueChangeStmt); // 记录value改变语句 // field_tmp = field_tmp + addedValue
            stmts.add(changeFieldStmt); // 记录改变字段语句 // field = field_tmp
        }
        return stmts; // 返回待插入的语句
    }

    // 给定注入的操作类型，对Field variable进行故障注入
    private boolean injectFieldWithAction(Body b, SootField field, String action) {
        Chain<Unit> units = b.getUnits();  // 获取当前代码语句
        Iterator<Unit> unitIt = units.snapshotIterator(); // 获取代码语句的快照迭代器
        while (unitIt.hasNext()) {
            Unit tmp = unitIt.next();
            Iterator<ValueBox> valueBoxes = tmp.getUseAndDefBoxes().iterator();
            while (valueBoxes.hasNext()) {
                Value value = valueBoxes.next().getValue();
                // check whether SootField in Soot is in SingleTon mode for equals comparison
                if ((value instanceof FieldRef) && (((FieldRef) value).getField().equals(field))) { // 定位到目标Field，此时tmp指向目标Field
                    logger.debug(tmp.toString());
                    try {
                        List<Stmt> stmts = this.getFieldStatementsByAction(b, field, action); // 获取Value故障注入的代码语句
                        for (int i = 0; i < stmts.size(); i++) {
                            if (i == 0) {
                                units.insertBefore(stmts.get(i), tmp);// 首句在tmp前插入
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

    // 将body内的variable按照域分组返回
    private Map<String, List<Object>> getAllVariables(Body b) {
        Map<String, List<Object>> result = new HashMap<String, List<Object>>();// 声明结果Map
        List<Local> tmpPLocals = b.getParameterLocals();// 获取全部的Parameter变量
        List<Object> pLocals = new ArrayList<Object>();// 声明符合要求的，参数类型的variable集合
        Chain<Local> tmpLocals = b.getLocals();// 获取全部的Local变量
        List<Object> locals = new ArrayList<Object>();// 声明符合要求的，局部变量类型的variable集合
        Chain<SootField> tmpFields = b.getMethod().getDeclaringClass().getFields();// 通过函数声明找到类声明再找到所有的Field变量
        List<Object> fields = new ArrayList<Object>();// 声明符合要求的，字段类型的variable集合

        int i;
        for (i = 0; i < tmpPLocals.size(); i++) {// 遍历每一个Parameter
            Local local = tmpPLocals.get(i);// 获取该Parameter
            if (this.inTargetedTypes(local.getType().toString()) && (!local.getName().startsWith("$"))) {// 若该参数的类型在处理范围之内
                pLocals.add(local);// 在参数类型的variable集合中，添加该variable
            }
        }

        Iterator<Local> tmpLocalsItr = tmpLocals.iterator();// 获取Local变量迭代器
        while (tmpLocalsItr.hasNext()) {// 遍历每一个Local
            Local local = tmpLocalsItr.next();// 获取该Local
            if (!pLocals.contains(local)) {// 如果该Local不是parameter
                if (this.inTargetedTypes(local.getType().toString()) && (!local.getName().startsWith("$"))) {// 若该参数的类型在处理范围之内
                    locals.add(local);// 在局部变量类型的variable集合中，添加该variable
                }
            }
        }
        // actually soot cannot get Class Fields
        Iterator<SootField> tmpFieldItr = tmpFields.iterator();// 获取Field变量迭代器
        while (tmpFieldItr.hasNext()) {// 遍历每一个Field
            SootField field = tmpFieldItr.next();// 获取该Field
            if (this.inTargetedTypes(field.getType().toString()) && (!field.getName().startsWith("$"))) {// 若该参数的类型在处理范围之内
                if (((field.getModifiers() & Modifier.FINAL) ^ Modifier.FINAL) == 0) {// not sure
                    // this is a final field, we cannot change value of it
//					logger.info(field.getDeclaration());
                } else {
                    fields.add(field);// 在字段类型的variable集合中，添加该variable
                }

            }
        }
        result.put("parameter", pLocals);// 在结果Map中添加符合要求的parameter类variable
        result.put("local", locals);// 在结果Map中添加符合要求的local类variable
        result.put("field", fields);// 在结果Map中添加符合要求的field类variable
        return result;// 返回结果Map
    }

    // 从配置文件的字符串，获取待注入variable的域
    private List<String> getTargetScope(String variableScope) {
        List<String> scopes = new ArrayList<String>();// 声明待注入域集合列表
        if ((variableScope == null) || (variableScope == "")) {// 若未指定待注入域
            scopes.add("local");// 待注入域集合添加"local"类
            scopes.add("field");// 待注入域集合添加"field"类
            scopes.add("parameter");// 待注入域集合添加"parameter"类
        } else {// 若指定了待注入域
            scopes.add(variableScope);// 直接在注入域集合添加指定的域类型
        }
        return scopes;
    }

    // 获取待注入variable的类别
    private List<String> getTargetType(String variableType) {
        List<String> types = new ArrayList<String>();// 声明待注入variable的类别集合列表
        if ((variableType == null) || (variableType == "")) {// 若未指定待注入variable的类别
            types.add("byte");// 待注入variable的类别集合添加"byte"类
            types.add("short");// 待注入variable的类别集合添加"short"类
            types.add("int");// 待注入variable的类别集合添加"int"类
            types.add("long");// 待注入variable的类别集合添加"long"类
            types.add("float");// 待注入variable的类别集合添加"float"类
            types.add("double");// 待注入variable的类别集合添加"double"类
            types.add("java.lang.String");// 待注入variable的类别集合添加"java.lang.String"类
        } else {
            if (variableType.equals("string")) {// 若指定的待注入variable类型为"string"
                types.add("java.lang.String");// 待注入variable的类别集合添加"java.lang.String"类
            } else {// 若指定的待注入variable类型为其他类
                types.add(variableType);// 直接在待注入variable的类别集合添加该类型
            }
        }
        return types;// 返回待注入variable的类别集合
    }

    // 判断给定的typeName是否在可以处理的7种类别之内
    private boolean inTargetedTypes(String typeName) {
        String[] targetTypes = {"byte", "short", "int", "long", "float", "double", "java.lang.String"};// 可以处理的的7种variable类型
        List<String> list = Arrays.asList(targetTypes);// 将String数组转换为String的List
        return list.contains(typeName);// 判断给定的typeName是否在可以处理的7种类别之内
    }

}
