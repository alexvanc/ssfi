package com.alex.ssfi;

import com.alex.ssfi.util.RunningParameter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.*;
import soot.util.Chain;

import java.util.*;

public class ShadowingAttributeTransformer extends BasicTransformer {//lhy 这个没有用到，那么
	private final Logger logger = LogManager.getLogger(InvokeRemovedTransformer.class);

	public ShadowingAttributeTransformer(RunningParameter parameters) {
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
		this.injectInfo.put("FaultType", "ATTRIBUTE_SHADOWED_FAULT");// 标记故障类型为，"ATTRIBUTE_SHADOWED_FAULT"
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());// 获取目标包名
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());// 获取目标类名
		this.injectInfo.put("Method", targetMethod.getSubSignature());// 获取目标函数签名

		logger.debug("Try to inject ATTRIBUTE_SHADOWED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
		List<String> actions = this.getTargetAction(this.parameters.getAction()); // 获取注入操作集合
		while (true) {
			int actionSize = actions.size();
			if (actionSize == 0) {
				break;
			}
			int targetActionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);
			String action = actions.get(targetActionIndex);// 获取注入操作
			actions.remove(targetActionIndex);
			// finally perform injection
			if (this.inject(b, action)) {// 进行注入
				this.parameters.setInjected(true);// 标记注入成功
				this.recordInjectionInfo();// 记录注入信息
				logger.debug("Succeed to inject ATTRIBUTE_SHADOWED_FAULT into " + this.injectInfo.get("Package") + " "
						+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
				return;
			} else {
				logger.debug("Failed injection:+ " + this.formatInjectionInfo());
			}
		}

		logger.debug("Fail to inject ATTRIBUTE_SHADOWED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

	}

	private synchronized SootMethod generateTargetMethod(Body b) {
		if (this.allQualifiedMethods == null) {
			this.initAllQualifiedMethods(b);
		}
		int leftQualifiedMethodsSize = this.allQualifiedMethods.size();
		if (leftQualifiedMethodsSize == 0) {
			return null;
		}
		int randomMethodIndex = new Random(System.currentTimeMillis()).nextInt(leftQualifiedMethodsSize);
		SootMethod targetMethod = this.allQualifiedMethods.get(randomMethodIndex);
		this.allQualifiedMethods.remove(randomMethodIndex);
		return targetMethod;
	}

	// at least there is one local with the same name with a field
	private void initAllQualifiedMethods(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();
		List<String> fieldsName = new ArrayList<String>();
		Iterator<SootField> allFields = b.getMethod().getDeclaringClass().getFields().snapshotIterator();
		while (allFields.hasNext()) {
			SootField field = allFields.next();
			fieldsName.add(field.getName());
		}
		boolean withSpefcifiedMethod = true;
		String specifiedMethodName = this.parameters.getMethodName();
		if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {
			withSpefcifiedMethod = false;
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
			Iterator<Local> locals = tmpBody.getLocals().snapshotIterator();

			while (locals.hasNext()) {
				Local local = locals.next();
				if (fieldsName.contains(local.getName())) {
					if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>"))&& (!method.getName().contains("<clinit>"))) {
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
	private boolean inject(Body b, String action) {
		this.injectInfo.put("Action", action);

		try {
			if (action.equals("field2local")) {// 若待注入操作为"field2local"

				List<FieldWithLocal> allTuples = this.getAllUsedFieldStmt(b); // 获取全部使用了Field的语句
				while (true) {
					int stmtSize = allTuples.size();
					if (stmtSize == 0) {
						break;
					}

					// 选取待注入的Field使用语句
					int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
					FieldWithLocal targetTuple = allTuples.get(stmtIndex);
					allTuples.remove(stmtIndex);

					if (this.injectLocalShadowField(b, targetTuple)) {// 注入field2local故障
						return true;
					}

				}

			} else if (action.equals("local2field")) {// 若待注入操作为"local2field"

				List<FieldWithLocal> allTuples = this.getAllUsedLocalStmt(b);// 获取全部使用了Local的语句
				while (true) {
					int stmtSize = allTuples.size();
					if (stmtSize == 0) {
						break;
					}

					// 选取待注入的Local使用语句
					int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
					FieldWithLocal targetTuple = allTuples.get(stmtIndex);
					allTuples.remove(stmtIndex);

					if (this.injectFieldShadowLocal(b, targetTuple)) {// 注入local2field故障
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

	// 获取注入操作集合
	private List<String> getTargetAction(String action) {
		List<String> actions = new ArrayList<String>();
		if ((action == null) || (action == "")) {
			actions.add("field2local");// 待注入操作集合 增加field2local操作
			actions.add("local2field");// 待注入操作集合 增加local2field操作
		} else {
			actions.add(action);// 待注入操作集合 增加指定操作
		}
		return actions;
	}

	// 注入local2field故障
	private boolean injectFieldShadowLocal(Body b, FieldWithLocal targetTuple) {
		Chain<Unit> units = b.getUnits();
		Local local = targetTuple.getLocal();
		SootField field = targetTuple.getField();
		Local fieldLocal = targetTuple.getFieldLocal();
		List<Stmt> stmts = targetTuple.getStmts();
		while (true) {
			int stmtSize = stmts.size();
			if (stmtSize == 0) {
				break;
			}

			int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
			Stmt targetStmt = stmts.get(stmtIndex); // 获取待注入的目标语句
			Unit nextUnit = units.getSuccOf(targetStmt);
			stmts.remove(stmtIndex);
			Stmt clonedStmt = (Stmt) targetStmt.clone();
			List<ValueBox> usedValueBoxes = clonedStmt.getUseBoxes();
			for (int j = 0, size = usedValueBoxes.size(); j < size; j++) {
				ValueBox box = usedValueBoxes.get(j);
				Value value = box.getValue();
				// we assume soot adopts singleton for each local, sootfield and unit
				if ((value instanceof Local) && (value.equals(local))) {

					if (box.canContainValue(fieldLocal)) {
						AssignStmt copyFieldToLocal;
						boolean isStaticField = false;
						if (((field.getModifiers() & Modifier.STATIC) ^ Modifier.STATIC) == 0) {
							isStaticField = true;
							this.injectInfo.put("static", "true");
						}
						if (isStaticField) {
							copyFieldToLocal = Jimple.v().newAssignStmt(fieldLocal,
									Jimple.v().newStaticFieldRef(field.makeRef())); // fieldLocal = field
						} else {
							// if this method is a static method, we try to replace the local with a
							// instanceField, error reported
							copyFieldToLocal = Jimple.v().newAssignStmt(fieldLocal,
									Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef())); // fieldLocal = field
						}
						units.insertBefore(copyFieldToLocal, targetStmt); // fieldLocal = field
						box.setValue(fieldLocal); // 将Local的使用改为Field的使用
						units.insertBefore(clonedStmt, targetStmt); // 使用fieldLocal

						List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), copyFieldToLocal);// 首句在copyFieldToLocal前插入
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
						for (int i = 0, actStmtSize = actStmts.size(); i < actStmtSize; i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));// 首句在conditionStmts[-1]后插入
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));// 之后每句在前一句之后插入
							}
						}

						GotoStmt skipInvoke = Jimple.v().newGotoStmt(nextUnit);// goto语句 goto nextUnit，跳过原Field使用语句
						units.insertAfter(skipInvoke, clonedStmt);
						return true;
					}

				}
			}

		}
		return false;

	}

	// 注入fiel2locald故障
	private boolean injectLocalShadowField(Body b, FieldWithLocal targetTuple) {
		Chain<Unit> units = b.getUnits();
		Local local = targetTuple.getLocal();
		Local fieldLocal = targetTuple.getFieldLocal();
		List<Stmt> stmts = targetTuple.getStmts();
		while (true) {
			int stmtSize = stmts.size();
			if (stmtSize == 0) {
				break;
			}
			int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
			Stmt targetStmt = stmts.get(stmtIndex);// 获取待注入的目标语句
			Unit nextUnit = units.getSuccOf(targetStmt);
			stmts.remove(stmtIndex);
			Stmt clonedStmt = (Stmt) targetStmt.clone();
			List<ValueBox> usedValueBoxes = clonedStmt.getUseBoxes();
			for (int j = 0, size = usedValueBoxes.size(); j < size; j++) {
				ValueBox box = usedValueBoxes.get(j);
				Value value = box.getValue();
				// we assume soot adopts singleton for each local, sootfield and unit
				if ((value instanceof Local) && (value.equals(fieldLocal))) {

					if (box.canContainValue(local)) {
						box.setValue(local);// 将Field的使用改为Local的使用
						units.insertBefore(clonedStmt, targetStmt);// 使用local
						List<Stmt> preStmts = this.getPrecheckingStmts(b);// 获取检查语句，检测激活模式，
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), clonedStmt);// 首句在clonedStmt前插入
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
						for (int i = 0, actStmtSize = actStmts.size(); i < actStmtSize; i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));// 首句在conditionStmts[-1]后插入
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));// 之后每句在前一句之后插入
							}
						}
						GotoStmt skipInvoke = Jimple.v().newGotoStmt(nextUnit);
						units.insertAfter(skipInvoke, clonedStmt);
						return true;
					}

				}
			}
		}
		return false;
	}

	// 获取全部Filed使用语句
	private List<FieldWithLocal> getAllUsedFieldStmt(Body b) {
		List<FieldWithLocal> allTuples = new ArrayList<FieldWithLocal>();
		Iterator<SootField> fieldItr = b.getMethod().getDeclaringClass().getFields().snapshotIterator();// 获取所有的Field

		while (fieldItr.hasNext()) {
			SootField field = fieldItr.next();

			Local fieldLocal = this.getFieldLocal(b, field);
			if (fieldLocal == null) {
				continue;
			}
			// this field is assigned to a local in this method and used
			Iterator<Local> localItr = b.getLocals().snapshotIterator();
			while (localItr.hasNext()) {
				Local local = localItr.next();
				String name = local.getName();
				if (!name.startsWith("$")) {
					// found a local with the same name and type of this field
					if (name.equals(field.getName())
							&& field.getType().toString().contentEquals(local.getType().toString())) {
						List<Stmt> stmts = this.getAllStmtsUsingLocal(b, fieldLocal); // 获取fieldLocal的全部使用语句
						if (stmts.size() != 0) {
							FieldWithLocal tuple = new FieldWithLocal();
							tuple.setField(field); // 记录Field变量
							tuple.setFieldLocal(fieldLocal);// 记录Field在局部变量中的代理（fieldLocal=field）
							tuple.setLocal(local); // 记录Local变量
							tuple.setStmts(stmts); // 记录fieldLocal的使用语句
							allTuples.add(tuple);
						}
						break;
					}
				}
			}

		}

		return allTuples;
	}

	// Actually every field is first assigned to a local variable
	// so we just need to find the local variable
	private List<FieldWithLocal> getAllUsedLocalStmt(Body b) {// 获取全部Local使用语句
		List<FieldWithLocal> allTuples = new ArrayList<FieldWithLocal>();

		Iterator<Local> localItr = b.getLocals().snapshotIterator();
		while (localItr.hasNext()) { // 对于每一个可能的local
			Local local = localItr.next();
			try {
				SootField field = b.getMethod().getDeclaringClass().getField(local.getName(), local.getType());
				List<Stmt> stmts = this.getAllStmtsUsingLocal(b, local); // 获取全部的Local使用语句
				FieldWithLocal tuple = new FieldWithLocal();
				Local fieldLocal = this.getFieldLocal(b, field);//获取field对应的local
				if (stmts.size() == 0) {
					continue;
				}
				if (fieldLocal == null) {
					Local fieldProxy = Jimple.v().newLocal("soot" + field.getName() + "Proxy", field.getType());
					b.getLocals().add(fieldProxy);
				}
				tuple.setField(field);// 记录field变量
				tuple.setLocal(local);// 记录local变量
				tuple.setStmts(stmts);// local的使用语句
				tuple.setFieldLocal(fieldLocal); // 记录Field在局部变量中的代理（fieldLocal=field）
				allTuples.add(tuple);
			} catch (Exception e) {
				// cannot found corresponding field
				continue;
			}
		}
		return allTuples;

	}

	// 获取local变量的全部使用语句
	private List<Stmt> getAllStmtsUsingLocal(Body b, Local local) {
		List<Stmt> stmts = new ArrayList<Stmt>();
		Iterator<Unit> unitItr = b.getUnits().snapshotIterator();
		while (unitItr.hasNext()) {
			Stmt stmt = (Stmt) unitItr.next();
			List<ValueBox> usedBoxes = stmt.getUseBoxes();
			for (int i = 0, size = usedBoxes.size(); i < size; i++) {
				Value usedValue = usedBoxes.get(i).getValue();
				if ((usedValue instanceof Local) && (usedValue.equals(local))) {
					stmts.add(stmt);
					break;
				}
			}
		}
		return stmts;
	}

	// 获取Field对应Local
	private Local getFieldLocal(Body b, SootField field) {
		List<Local> allPossibleLocals = new ArrayList<Local>();
		// actually a field could be assign to many different locals
		// here we just random return one local
		Iterator<Unit> unitItr = b.getUnits().snapshotIterator();
		while (unitItr.hasNext()) {
			Unit unit = unitItr.next();
			if (unit instanceof AssignStmt) {
				// left is a local, right is a sootfieldRef
				AssignStmt targetStmt = (AssignStmt) unit;
				Value leftValue = targetStmt.getLeftOp();
				Value rightValue = targetStmt.getRightOp();
				if ((leftValue instanceof Local) && (rightValue instanceof FieldRef)) {
					SootField rightField = ((FieldRef) rightValue).getField();
					if (rightField.equals(field)) {
						allPossibleLocals.add((Local) leftValue);
					}
				}
			}
		}
		int size = allPossibleLocals.size();
		if (size == 0) {
			return null;
		} else {
			return allPossibleLocals.get(new Random(System.currentTimeMillis()).nextInt(size));
		}
	}

	// 连接Field变量、Local变量、Field在局部的代理变量（fieldLocal=field）、fieldLocal或Local的使用语句集
	class FieldWithLocal {
		private SootField field;
		private Local local;
		private Local fieldLocal;
		private List<Stmt> stmts;

		public FieldWithLocal() {

		}

		public SootField getField() {
			return field;
		}

		public void setField(SootField field) {
			this.field = field;
		}

		public Local getLocal() {
			return local;
		}

		public void setLocal(Local local) {
			this.local = local;
		}

		public Local getFieldLocal() {
			return fieldLocal;
		}

		public void setFieldLocal(Local fieldLocal) {
			this.fieldLocal = fieldLocal;
		}

		public List<Stmt> getStmts() {
			return stmts;
		}

		public void setStmts(List<Stmt> stmts) {
			this.stmts = stmts;
		}
	}

}
