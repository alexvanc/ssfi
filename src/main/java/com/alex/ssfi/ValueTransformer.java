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

/*
 * TODO
 * distinguish between static field and instance field
 * report the Soot bugs of failing to get the fields of a class
 */
public class ValueTransformer extends BasicTransformer {

	private final Logger logger = LogManager.getLogger(ValueTransformer.class);

	public ValueTransformer(RunningParameter parameters) {
		super(parameters);

	}

	public ValueTransformer() {
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
			Body tmpBody;
			try {
				tmpBody = targetMethod.retrieveActiveBody();
			} catch (Exception e) {
				logger.info("Retrieve Active Body Failed!");
				continue;
			}
			if (tmpBody == null) {
				continue;
			}
			this.startToInject(tmpBody);
		}

	}

	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "Value_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());
		List<String> scopes = getTargetScope(this.parameters.getVariableScope());

		Map<String, List<Object>> allVariables = this.getAllVariables(b);
		logger.debug("Try to inject Value_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// randomly combine scope, type , variable, action
		while (true) {
			int scopesSize = scopes.size();
			if (scopesSize == 0) {
				logger.debug("Cannot find qualified scopes");
				break;
			}
			int scopeIndex = new Random(System.currentTimeMillis()).nextInt(scopesSize);
			String scope = scopes.get(scopeIndex);
			scopes.remove(scopeIndex);
			this.injectInfo.put("VariableScope", scope);
			List<String> types = getTargetType(this.parameters.getVariableType());
			// randomly combine type, variable, action
			while (true) {
				int typesSize = types.size();
				if (typesSize == 0) {
					break;
				}
				int typeIndex = new Random(System.currentTimeMillis()).nextInt(typesSize);
				String type = types.get(typeIndex);
				types.remove(typeIndex);
				this.injectInfo.put("VariableType", type);
				// randomly combine variable, action
				List<Object> qualifiedVariables = this.getQualifiedVariables(b, allVariables, scope, type);
				while (true) {
					int variablesSize = qualifiedVariables.size();
					if (variablesSize == 0) {
						break;
					}
					int variableIndex = new Random(System.currentTimeMillis()).nextInt(variablesSize);
					Object variable = qualifiedVariables.get(variableIndex);
					qualifiedVariables.remove(variableIndex);
					// randomly generate an action for this variable
					List<String> possibleActions = this.getPossibleActionsForVariable(type);
					while (true) {
						int actionSize = possibleActions.size();
						if (actionSize == 0) {
							break;
						}
						int actionIndex = new Random(System.currentTimeMillis()).nextInt(actionSize);
						String action = possibleActions.get(actionIndex);
						possibleActions.remove(actionIndex);
						this.injectInfo.put("Action", action);
						// finally perform injection
						if (this.inject(b, scope, variable, action)) {
							this.parameters.setInjected(true);
							this.recordInjectionInfo();
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

	private boolean inject(Body b, String scope, Object variable, String action) {
		if (scope.equals("field")) {// inject fault to field variables
			SootField field = (SootField) variable;
			this.injectInfo.put("VariableName", field.getName());
			this.injectInfo.put("Action", action);
			if (this.injectFieldWithAction(b, field, action)) {
				return true;
			}
		} else if (scope.equals("local")) {// inject this fault to local or parameter variables
			Local local = (Local) variable;
			this.injectInfo.put("VariableName", local.getName());
			this.injectInfo.put("Action", action);
			if (this.injectLocalWithAction(b, local, action)) {
				return true;
			}
		} else if (scope.equals("parameter")) {
			Local local = (Local) variable;
			this.injectInfo.put("VariableName", local.getName());
			this.injectInfo.put("Action", action);
			if (this.injectParameterWithAction(b, local, action)) {
				return true;
			}
		}
		return false;
	}

	private List<String> getPossibleActionsForVariable(String type) {
		// For simplicity, we don't check whether the action is applicable to this type
		// of variables
		// this should be done in the configuration validator
		List<String> possibleActions = new ArrayList<String>();
		String specifiedAction = this.parameters.getAction();
		String specifiedValue = this.parameters.getVariableValue();
		if ((specifiedAction == null) || (specifiedAction == "")) {
			if (type.equals("java.lang.String")) {
				possibleActions.add("TO");
			} else {
				if ((specifiedValue != null) && (specifiedValue != "")) {
					possibleActions.add("ADD");
					possibleActions.add("SUB");
					possibleActions.add("TO");
				} else {
					possibleActions.add("ADD");
					possibleActions.add("SUB");
				}

			}
		} else {
			possibleActions.add(specifiedAction);
		}
		return possibleActions;
	}

	private List<Object> getQualifiedVariables(Body b, Map<String, List<Object>> allVariables, String scope,
			String type) {
		List<Object> qualifiedVariables = new ArrayList<Object>();
		List<Object> variablesInScope = allVariables.get(scope);
		int length = variablesInScope.size();
		String targetVariableName = this.parameters.getVariableName();

		if (scope.equals("local") || scope.equals("parameter")) {
			for (int i = 0; i < length; i++) {
				Local localVariable = (Local) variablesInScope.get(i);
				if (localVariable.getType().toString().equals(type)) {
					// if users specify the variableName
					if ((targetVariableName == null) || (targetVariableName == "")) {
						qualifiedVariables.add(localVariable);
					} else {
						if (localVariable.getName().equals(targetVariableName)) {
							qualifiedVariables.add(localVariable);
						}
					}
				}
			}
		} else {// scope is field
			for (int i = 0; i < length; i++) {
				SootField fieldVariable = (SootField) variablesInScope.get(i);
				if (fieldVariable.getType().toString().equals(type)) {
					// if users specify the variableName
					if ((targetVariableName == null) || (targetVariableName == "")) {
						qualifiedVariables.add(fieldVariable);
					} else {
						if (fieldVariable.getName().equals(targetVariableName)) {
							qualifiedVariables.add(fieldVariable);
						}
					}
				}
			}
		}
		return qualifiedVariables;
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

	// for this fault type,we simply assume all methods satisfy the condition
	private void initAllQualifiedMethods(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		List<SootMethod> allQualifiedMethods = new ArrayList<SootMethod>();
		boolean withSpefcifiedMethod = true;
		String specifiedMethodName = this.parameters.getMethodName();
		if ((specifiedMethodName == null) || (specifiedMethodName.equals(""))) {
			withSpefcifiedMethod = false;
		}
		int length = allMethods.size();
		for (int i = 0; i < length; i++) {
			SootMethod method = allMethods.get(i);

			if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>"))&& (!method.getName().contains("<clinit>"))) {
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

	private boolean injectLocalWithAction(Body b, Local local, String action) {
		// inject at the beginning or inject after assignment make a lot of difference
		// currently we inject the fault after it is visited once
		logger.info(this.formatInjectionInfo());
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitIt = units.snapshotIterator();
		while (unitIt.hasNext()) {
			Unit tmp = unitIt.next();
			Iterator<ValueBox> valueBoxes = tmp.getUseBoxes().iterator();
			while (valueBoxes.hasNext()) {
				Value value = valueBoxes.next().getValue();
				if ((value instanceof Local) && (value.equivTo(local))) {
					logger.debug(tmp.toString());
					try {
						List<Stmt> stmts = this.getLocalStatementsByAction(local, action);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								// here is the difference between local and parameter variable
								units.insertBefore(stmts.get(i), tmp);
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}
						List<Stmt> preStmts = this.getPrecheckingStmts(b);
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), stmts.get(0));
							} else {
								units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
							}
						}
						List<Stmt> conditionStmts = this.getConditionStmt(b, tmp);
						for (int i = 0; i < conditionStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
							} else {
								units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
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

	private boolean injectParameterWithAction(Body b, Local local, String action) {
		// inject at the beginning or inject after assignment make a lot of difference
		// currently we inject the fault after it is visited once
		logger.info(this.formatInjectionInfo());
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitIt = units.snapshotIterator();
		while (unitIt.hasNext()) {
			Unit tmp = unitIt.next();
			Iterator<ValueBox> valueBoxes = tmp.getUseAndDefBoxes().iterator();
			while (valueBoxes.hasNext()) {
				Value value = valueBoxes.next().getValue();
				if ((value instanceof Local) && (value.equivTo(local))) {
					logger.debug(tmp.toString());
					Unit nextUnit = units.getSuccOf(tmp);
					try {
						List<Stmt> stmts = this.getLocalStatementsByAction(local, action);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(stmts.get(i), tmp);
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}
						List<Stmt> preStmts = this.getPrecheckingStmts(b);
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), stmts.get(0));
							} else {
								units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
							}
						}
						List<Stmt> conditionStmts = this.getConditionStmt(b, nextUnit);
						for (int i = 0; i < conditionStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
							} else {
								units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
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

	private List<Stmt> getLocalStatementsByAction(Local local, String action) {
		Stmt valueChangeStmt = null;
		if (action.equals("TO")) {
			String typeName = local.getType().toString();
			String targetStringValue = this.parameters.getVariableValue();
			if (targetStringValue != null) {
				this.injectInfo.put("VariableValue", targetStringValue);
			}

			if (typeName.equals("byte")) {
				byte result = Byte.parseByte(targetStringValue);
				valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result));
			} else if (typeName.equals("short")) {
				short result = Short.parseShort(targetStringValue);
				valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result));
			} else if (typeName.equals("int")) {
				int result = Integer.parseInt(targetStringValue);
				valueChangeStmt = Jimple.v().newAssignStmt(local, IntConstant.v(result));
			} else if (typeName.equals("long")) {
				long result = Long.parseLong(targetStringValue);
//				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
				valueChangeStmt = Jimple.v().newAssignStmt(local, LongConstant.v(result));
			} else if (typeName.equals("float")) {
				float result = Float.parseFloat(targetStringValue);
				valueChangeStmt = Jimple.v().newAssignStmt(local, FloatConstant.v(result));
			} else if (typeName.equals("double")) {
				double result = Double.parseDouble(targetStringValue);
				valueChangeStmt = Jimple.v().newAssignStmt(local, DoubleConstant.v(result));
//			}else if(typeName.equals("bool")) {
//				boolean result=Boolean.parseBoolean(targetStringValue);
////				valueChangeStmt=Jimple.v().newAssignStmt(local, Boolean.v(result));
//				Jimple.v().new
			} else if (typeName.equals("java.lang.String")) {
				if ((targetStringValue == null) || (targetStringValue == "")) {
					targetStringValue = "test";
				}
				valueChangeStmt = Jimple.v().newAssignStmt(local, StringConstant.v(targetStringValue));
			}
		} else if (action.equals("ADD")) {
			String typeName = local.getType().toString();
			String addedStringValue = this.parameters.getVariableValue();
			if (addedStringValue != null) {
				this.injectInfo.put("VariableValue", addedStringValue);
			}
			if (typeName.equals("byte")) {
				byte addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Byte.parseByte(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
			} else if (typeName.equals("short")) {
				short addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Short.parseShort(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("int")) {
				int addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Integer.parseInt(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("long")) {
				long addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Long.parseLong(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, LongConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("float")) {
				float addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Float.parseFloat(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
			} else if (typeName.equals("double")) {
				double addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Double.parseDouble(addedStringValue);
				}
				AddExpr addExp = Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
//			}else if(typeName.equals("bool")) {
//				
			}

		} else if (action.equals("SUB")) {
			String typeName = local.getType().toString();
			String subedStringValue = this.parameters.getVariableValue();
			if (subedStringValue != null) {
				this.injectInfo.put("VariableValue", subedStringValue);
			}

			if (typeName.equals("byte")) {
				byte addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Byte.parseByte(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
			} else if (typeName.equals("short")) {
				short addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Short.parseShort(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("int")) {
				int addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Integer.parseInt(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("long")) {
				long addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Long.parseLong(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, LongConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);

			} else if (typeName.equals("float")) {
				float addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Float.parseFloat(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
			} else if (typeName.equals("double")) {
				double addedValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					addedValue = Double.parseDouble(subedStringValue);
				}
				SubExpr addExp = Jimple.v().newSubExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
//			}else if(typeName.equals("bool")) {
//				
			}
		}
		List<Stmt> stmts = new ArrayList<Stmt>();
		stmts.add(valueChangeStmt);
		return stmts;
	}

	private List<Stmt> getFieldStatementsByAction(Body b, SootField field, String action) {
		// TODO
		// verify the field API bugs
		Stmt valueChangeStmt = null;
		Stmt copyFieldStmt = null;
		Stmt changeFieldStmt = null;
		boolean isStaticField = false;
		if (((field.getModifiers() & Modifier.STATIC) ^ Modifier.STATIC) == 0) {
			isStaticField = true;
			this.injectInfo.put("static", "true");
		}

		List<Stmt> stmts = new ArrayList<Stmt>();
		Local local = null;
		if (action.equals("TO")) {
			String typeName = field.getType().toString();
			String targetStringValue = this.parameters.getVariableValue();
			if (targetStringValue != null) {
				this.injectInfo.put("VariableValue", targetStringValue);
			}

			if (typeName.equals("byte")) {
//				
				byte result = Byte.parseByte(targetStringValue);
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							IntConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result));
				}
			} else if (typeName.equals("short")) {
				short result = Short.parseShort(targetStringValue);
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							IntConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result));
				}

			} else if (typeName.equals("int")) {
				int result = Integer.parseInt(targetStringValue);
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							IntConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), IntConstant.v(result));
				}

			} else if (typeName.equals("long")) {
				long result = Long.parseLong(targetStringValue);
//				AddExpr addExp=Jimple.v().newAddExpr(target, LongConstant.v(1));
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							LongConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), LongConstant.v(result));
				}
			} else if (typeName.equals("float")) {
				float result = Float.parseFloat(targetStringValue);
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							FloatConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), FloatConstant.v(result));
				}
			} else if (typeName.equals("double")) {
				double result = Double.parseDouble(targetStringValue);
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							DoubleConstant.v(result));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
							DoubleConstant.v(result));
				}
//			}else if(typeName.equals("bool")) {
//				boolean result=Boolean.parseBoolean(targetStringValue);
////				valueChangeStmt=Jimple.v().newAssignStmt(local, Boolean.v(result));
//				Jimple.v().new
			} else if (typeName.equals("java.lang.String")) {
				if ((targetStringValue == null) || (targetStringValue == "")) {
					targetStringValue = "test";
				}
				if (isStaticField) {
					valueChangeStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()),
							StringConstant.v(targetStringValue));
				} else {
					valueChangeStmt = Jimple.v().newAssignStmt(
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
							StringConstant.v(targetStringValue));
				}
			}
			stmts.add(valueChangeStmt);
		} else if (action.equals("ADD")) {
			String typeName = field.getType().toString();
			String addedStringValue = this.parameters.getVariableValue();
			if (addedStringValue != null) {
				this.injectInfo.put("VariableValue", addedStringValue);
			}
			if (typeName.equals("byte")) {
				byte addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Byte.parseByte(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));
					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}
			} else if (typeName.equals("short")) {
				short addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Short.parseShort(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", ShortType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));
					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("int")) {
				int addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Integer.parseInt(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", IntType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, IntConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}
			} else if (typeName.equals("long")) {
				long addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Long.parseLong(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", LongType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, LongConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("float")) {
				float addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Float.parseFloat(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", FloatType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, FloatConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("double")) {
				double addedValue = 1;
				if ((addedStringValue != null) && (addedStringValue != "")) {
					addedValue = Double.parseDouble(addedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", DoubleType.v());
				b.getLocals().add(local);
				AddExpr addExp = Jimple.v().newAddExpr(local, DoubleConstant.v(addedValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

//			}else if(typeName.equals("bool")) {
//				
			}
			stmts.add(copyFieldStmt);
			stmts.add(valueChangeStmt);
			stmts.add(changeFieldStmt);
		} else if (action.equals("SUB")) {
			String typeName = field.getType().toString();
			String subedStringValue = this.parameters.getVariableValue();
			if (subedStringValue != null) {
				this.injectInfo.put("VariableValue", subedStringValue);
			}
			if (typeName.equals("byte")) {
				byte subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Byte.parseByte(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", ByteType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				}

			} else if (typeName.equals("short")) {
				short subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Short.parseShort(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", ShortType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));
					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("int")) {
				int subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Integer.parseInt(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", IntType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, IntConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));
					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("long")) {
				long subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Long.parseLong(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", LongType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, LongConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}
			} else if (typeName.equals("float")) {
				float subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Float.parseFloat(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", FloatType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, FloatConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));
					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));
					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

			} else if (typeName.equals("double")) {
				double subededValue = 1;
				if ((subedStringValue != null) && (subedStringValue != "")) {
					subededValue = Double.parseDouble(subedStringValue);
				}
				local = Jimple.v().newLocal("field_tmp", DoubleType.v());
				b.getLocals().add(local);
				SubExpr addExp = Jimple.v().newSubExpr(local, DoubleConstant.v(subededValue));
				valueChangeStmt = Jimple.v().newAssignStmt(local, addExp);
				if (isStaticField) {
					copyFieldStmt = Jimple.v().newAssignStmt(local, Jimple.v().newStaticFieldRef(field.makeRef()));

					changeFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), local);
				} else {
					copyFieldStmt = Jimple.v().newAssignStmt(local,
							Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()));

					changeFieldStmt = Jimple.v()
							.newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()), local);
				}

//			}else if(typeName.equals("bool")) {
			}
			stmts.add(copyFieldStmt);
			stmts.add(valueChangeStmt);
			stmts.add(changeFieldStmt);
		}
		return stmts;
	}

	private boolean injectFieldWithAction(Body b, SootField field, String action) {
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitIt = units.snapshotIterator();
		while (unitIt.hasNext()) {
			Unit tmp = unitIt.next();
			Iterator<ValueBox> valueBoxes = tmp.getUseAndDefBoxes().iterator();
			while (valueBoxes.hasNext()) {
				Value value = valueBoxes.next().getValue();
				// TODO
				// check whether SootField in Soot is in SingleTon mode for equals comparison
				if ((value instanceof FieldRef) && (((FieldRef) value).getField().equals(field))) {
					logger.debug(tmp.toString());
					try {
						List<Stmt> stmts = this.getFieldStatementsByAction(b, field, action);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(stmts.get(i), tmp);
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}
						List<Stmt> preStmts = this.getPrecheckingStmts(b);
						for (int i = 0; i < preStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(preStmts.get(i), stmts.get(0));
							} else {
								units.insertAfter(preStmts.get(i), preStmts.get(i - 1));
							}
						}
						List<Stmt> conditionStmts = this.getConditionStmt(b, tmp);
						for (int i = 0; i < conditionStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(conditionStmts.get(i), preStmts.get(preStmts.size() - 1));
							} else {
								units.insertAfter(conditionStmts.get(i), conditionStmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
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

	private Map<String, List<Object>> getAllVariables(Body b) {
		Map<String, List<Object>> result = new HashMap<String, List<Object>>();
		List<Local> tmpPLocals = b.getParameterLocals();
		List<Object> pLocals = new ArrayList<Object>();
		Chain<Local> tmpLocals = b.getLocals();
		List<Object> locals = new ArrayList<Object>();
		Chain<SootField> tmpFields = b.getMethod().getDeclaringClass().getFields();
		List<Object> fields = new ArrayList<Object>();

		int i;
		for (i = 0; i < tmpPLocals.size(); i++) {
			Local local = tmpPLocals.get(i);
			if (this.inTargetedTypes(local.getType().toString()) && (!local.getName().startsWith("$"))) {
				pLocals.add(local);
			}
		}

		Iterator<Local> tmpLocalsItr = tmpLocals.iterator();
		while (tmpLocalsItr.hasNext()) {
			Local local = tmpLocalsItr.next();
			if (!pLocals.contains(local)) {
				if (this.inTargetedTypes(local.getType().toString()) && (!local.getName().startsWith("$"))) {
					locals.add(local);
				}
			}

		}

		// actually soot cannot get Class Fields
		Iterator<SootField> tmpFieldItr = tmpFields.iterator();
		while (tmpFieldItr.hasNext()) {
			SootField field = tmpFieldItr.next();
//			logger.info("Field name:　" + field.getName() + " Field type: " + field.getType().toString());
			if (this.inTargetedTypes(field.getType().toString()) && (!field.getName().startsWith("$"))) {
				if (((field.getModifiers() & Modifier.FINAL) ^ Modifier.FINAL) == 0) {// not sure
					// this is a final field, we cannot change value of it
//					logger.info(field.getDeclaration());
				} else {
					fields.add(field);
				}

			}
		}
		result.put("parameter", pLocals);
		result.put("local", locals);
		result.put("field", fields);
		return result;
	}

//	private boolean performInjection(String scope,String type,String name,Body b) {
//		b.getP
//		if(scope.equals("field")) {
//			b.getMethod().getDeclaringClass().get
//		}
//		return false;
//	}
	private List<String> getTargetScope(String variableScope) {
		List<String> scopes = new ArrayList<String>();
		if ((variableScope == null) || (variableScope == "")) {
			scopes.add("local");
			scopes.add("field");
			scopes.add("parameter");
		} else {
			scopes.add(variableScope);
		}
		return scopes;
	}

	private List<String> getTargetType(String variableType) {
		List<String> types = new ArrayList<String>();
		if ((variableType == null) || (variableType == "")) {
			types.add("byte");
			types.add("short");
			types.add("int");
			types.add("long");
			types.add("float");
			types.add("double");
			types.add("java.lang.String");
//			types.add("bool");
		} else {
			if (variableType.equals("string")) {
				types.add("java.lang.String");
			} else {
				types.add(variableType);
			}
		}
		return types;
	}

	private boolean inTargetedTypes(String typeName) {
//		String[] targetTypes= {"boolean","byte","short","int","long","float","double","java.lang.String"};
		String[] targetTypes = { "byte", "short", "int", "long", "float", "double", "java.lang.String" };
		List<String> list = Arrays.asList(targetTypes);
		if (list.contains(typeName)) {
			return true;
		} else {
			return false;
		}
	}

}
