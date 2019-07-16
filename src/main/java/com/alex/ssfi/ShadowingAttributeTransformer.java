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
import soot.Modifier;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.ValueBox;
import soot.jimple.AssignStmt;
import soot.jimple.FieldRef;
import soot.jimple.GotoStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.util.Chain;

public class ShadowingAttributeTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(InvokeRemovedTransformer.class);

	public ShadowingAttributeTransformer(RunningParameter parameters) {
		this.parameters = parameters;

	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub

		while (!this.parameters.isInjected()) {
			// in this way, all the FIs are performed in the first function of this class
			SootMethod targetMethod = this.generateTargetMethod(b);
			if (targetMethod == null) {
				return;
			}
			this.startToInject(targetMethod.getActiveBody());
		}

	}

	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "ATTRIBUTE_SHADOWED_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());

		logger.debug("Try to inject ATTRIBUTE_SHADOWED_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
		List<String> actions = this.getTargetAction(this.parameters.getAction());
		while (true) {
			int actionSize = actions.size();
			if (actionSize == 0) {
				break;
			}
			int targetActionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);
			String action = actions.get(targetActionIndex);
			actions.remove(targetActionIndex);
			// finally perform injection
			if (this.inject(b, action)) {
				this.parameters.setInjected(true);
				this.recordInjectionInfo();
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
					if (!withSpefcifiedMethod) {
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

	private boolean inject(Body b, String action) {
		this.injectInfo.put("Action", action);

		try {
			if (action.equals("field2local")) {

				List<FieldWithLocal> allTuples = this.getAllUsedFieldStmt(b);
				while (true) {
					int stmtSize = allTuples.size();
					if (stmtSize == 0) {
						break;
					}
					int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
					FieldWithLocal targetTuple = allTuples.get(stmtIndex);
					allTuples.remove(stmtIndex);

					if (this.injectLocalShadowField(b, targetTuple)) {
						return true;
					}

				}

			} else if (action.equals("local2field")) {

				List<FieldWithLocal> allTuples = this.getAllUsedLocalStmt(b);
				while (true) {
					int stmtSize = allTuples.size();
					if (stmtSize == 0) {
						break;
					}
					int stmtIndex = new Random(System.currentTimeMillis()).nextInt(stmtSize);
					FieldWithLocal targetTuple = allTuples.get(stmtIndex);
					allTuples.remove(stmtIndex);
					if (this.injectFieldShadowLocal(b, targetTuple)) {
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

	private List<String> getTargetAction(String action) {
		List<String> actions = new ArrayList<String>();
		if ((action == null) || (action == "")) {
			actions.add("field2local");
			actions.add("local2field");
		} else {
			actions.add(action);
		}
		return actions;
	}

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
			Stmt targetStmt = stmts.get(stmtIndex);
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
									Jimple.v().newStaticFieldRef(field.makeRef()));
						} else {
							// if this method is a static method, we try to replace the local with a
							// instanceField, error reported
							copyFieldToLocal = Jimple.v().newAssignStmt(fieldLocal,
									Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
						}
						units.insertBefore(copyFieldToLocal, targetStmt);
						box.setValue(fieldLocal);
						units.insertBefore(clonedStmt, targetStmt);

						List<Stmt> preStmts = this.getPrecheckingStmts(b);
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), copyFieldToLocal);
							} else {
								units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
							}
						}
						List<Stmt> conditionStmts = this.getConditionStmt(b, targetStmt);
						for (int i = 0; i < conditionStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
							} else {
								units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0, actStmtSize = actStmts.size(); i < actStmtSize; i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
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
			Stmt targetStmt = stmts.get(stmtIndex);
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
						box.setValue(local);
						units.insertBefore(clonedStmt, targetStmt);
						List<Stmt> preStmts = this.getPrecheckingStmts(b);
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), clonedStmt);
							} else {
								units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
							}
						}
						List<Stmt> conditionStmts = this.getConditionStmt(b, targetStmt);
						for (int i = 0; i < conditionStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
							} else {
								units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
							}
						}
						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0, actStmtSize = actStmts.size(); i < actStmtSize; i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
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

	private List<FieldWithLocal> getAllUsedFieldStmt(Body b) {
		List<FieldWithLocal> allTuples = new ArrayList<FieldWithLocal>();
		Iterator<SootField> fieldItr = b.getMethod().getDeclaringClass().getFields().snapshotIterator();

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
						List<Stmt> stmts = this.getAllStmtsUsingLocal(b, fieldLocal);
						if (stmts.size() != 0) {
							FieldWithLocal tuple = new FieldWithLocal();
							tuple.setField(field);
							tuple.setFieldLocal(fieldLocal);
							tuple.setLocal(local);
							tuple.setStmts(stmts);
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
	private List<FieldWithLocal> getAllUsedLocalStmt(Body b) {
		List<FieldWithLocal> allTuples = new ArrayList<FieldWithLocal>();

		Iterator<Local> localItr = b.getLocals().snapshotIterator();
		while (localItr.hasNext()) {
			Local local = localItr.next();
			try {
				SootField field = b.getMethod().getDeclaringClass().getField(local.getName(), local.getType());
				List<Stmt> stmts = this.getAllStmtsUsingLocal(b, local);
				FieldWithLocal tuple = new FieldWithLocal();
				Local fieldLocal = this.getFieldLocal(b, field);
				if (stmts.size() == 0) {
					continue;
				}
				if (fieldLocal == null) {
					Local fieldProxy = Jimple.v().newLocal("soot" + field.getName() + "Proxy", field.getType());
					b.getLocals().add(fieldProxy);
				}
				tuple.setField(field);
				tuple.setLocal(local);
				tuple.setStmts(stmts);
				tuple.setFieldLocal(fieldLocal);
				allTuples.add(tuple);
			} catch (Exception e) {
				// cannot found corresponding field
				continue;
			}
		}
		return allTuples;

	}

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
