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

/*
 * TODO
 * distinguish between static field and instance field
 * decide whether to inject NullFault to the return value
 */
public class NullTransformer extends BasicTransformer {
	private final Logger logger = LogManager.getLogger(ValueTransformer.class);

	public NullTransformer(RunningParameter parameters) {
		super(parameters);

	}

	public NullTransformer() {
	}

	@Override
	protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
		// TODO Auto-generated method stub

		this.methodIndex++;
		if (this.parameters.isInjected()) {
			return;
		}
		String methodName = b.getMethod().getName();
		String methodSubSignature = b.getMethod().getSubSignature();
		String specifiedMethodName = this.parameters.getMethodName();
		if ((specifiedMethodName == null) || (specifiedMethodName == "")) {// in the random method mode
			if (!this.foundTargetMethod) {
				// randomly generate a target method
				this.generateTargetMethod(b);
			}
			if (methodSubSignature.equals(this.targetMethodSubSignature)) {
				this.startToInject(b);
			} else {
				return;
			}
		} else {// in the customized method mode
			if (methodName.equalsIgnoreCase(specifiedMethodName)) {
				this.startToInject(b);
			} else {
				return;
			}
		}

	}

	private void generateTargetMethod(Body b) {
		List<SootMethod> allMethods = b.getMethod().getDeclaringClass().getMethods();
		if (this.methodIndex >= allMethods.size()) {
			return;
		}
		int targetMethodIndex = new Random(System.currentTimeMillis())
				.nextInt(allMethods.size() - this.methodIndex + 1);
		this.foundTargetMethod = true;
		this.targetMethodSubSignature = allMethods.get(this.methodIndex + targetMethodIndex - 1).getSubSignature();
		return;
	}

	private void startToInject(Body b) {
		// no matter this inject fails or succeeds, this targetMethod is already used
		this.foundTargetMethod = false;
		SootMethod targetMethod = b.getMethod();
		this.injectInfo.put("FaultType", "NULL_FAULT");
		this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
		this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
		this.injectInfo.put("Method", targetMethod.getSubSignature());
		List<String> scopes = getTargetScope(this.parameters.getVariableScope());

		Map<String, List<Object>> allVariables = this.getAllVariables(b);
		logger.debug("Try to inject NULL_FAULT into " + this.injectInfo.get("Package") + " "
				+ this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

		// randomly combine scope, variable
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

			// randomly combine variable, action
			List<Object> qualifiedVariables = this.getQualifiedVariables(b, allVariables, scope);
			while (true) {
				int variablesSize = qualifiedVariables.size();
				if (variablesSize == 0) {
					break;
				}
				int variableIndex = new Random(System.currentTimeMillis()).nextInt(variablesSize);
				Object variable = qualifiedVariables.get(variableIndex);
				qualifiedVariables.remove(variableIndex);

				// finally perform injection
				if (this.inject(b, scope, variable)) {
					this.parameters.setInjected(true);
					this.recordInjectionInfo();
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

	private boolean inject(Body b, String scope, Object variable) {
		if (scope.equals("local")) {
			if (this.injectLocalWithNull(b, (Local) variable)) {

				return true;
			}
		} else if (scope.equals("parameter")) {
			// TODO
			// currently local and parameter are processed in the same way, decide later
			if (this.injectParameterWithNull(b, (Local) variable)) {
	
				return true;
			}
		} else if (scope.equals("field")) {
			if (this.injectFieldWithNull(b, (SootField) variable)) {

				return true;
			}
		} else if (scope.equals("return")) {
			if (this.injectReturnWithNull(b, (Stmt) variable)) {
	
				return true;
			}
		}
		return false;
	}

	private boolean injectFieldWithNull(Body b, SootField field) {
		this.injectInfo.put("VariableName", field.getName());
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
						List<Stmt> stmts = this.getNullFieldStatements(b, field);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								// here we add the field null to the begining of the method
								units.insertBefore(stmts.get(i), units.getFirst());
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), stmts.get(stmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
							}
						}
						return true;
					} catch (Exception e) {
						logger.error(e.getMessage());
						logger.error(this.injectInfo.toString());
						return false;
					}

				}
			}

		}
		return false;
	}

	private List<Stmt> getNullFieldStatements(Body b, SootField field) {
		Stmt nullFieldStmt = null;
		boolean isStaticField = false;
		if (((field.getModifiers() & Modifier.STATIC) ^ Modifier.STATIC) == 0) {
			isStaticField = true;
			this.injectInfo.put("static", "true");
		}

		if (isStaticField) {
			nullFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newStaticFieldRef(field.makeRef()), NullConstant.v());
		} else {
			nullFieldStmt = Jimple.v().newAssignStmt(Jimple.v().newInstanceFieldRef(b.getThisLocal(), field.makeRef()),
					NullConstant.v());
		}

		List<Stmt> stmts = new ArrayList<Stmt>();
		stmts.add(nullFieldStmt);
		return stmts;
	}

	private boolean injectLocalWithNull(Body b, Local local) {
		this.injectInfo.put("VariableName", local.getName());
		logger.debug(this.formatInjectionInfo());
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
						List<Stmt> stmts = this.getNullLocalStatements(local);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								// here is the difference between local and parameter variable
								units.insertBefore(stmts.get(i), tmp);
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), stmts.get(stmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
							}
						}
						return true;
					} catch (Exception e) {
						logger.error(e.getMessage());
						return false;
					}

				}
			}

		}
		return false;
	}

	private boolean injectReturnWithNull(Body b, Stmt stmt) {
		Chain<Unit> units = b.getUnits();
		Iterator<Unit> unitIt = units.snapshotIterator();
		while (unitIt.hasNext()) {
			Unit tmp = unitIt.next();
			if (tmp.equals(stmt)) {
				try {
					Stmt targetStmt = (Stmt) tmp;
					if (targetStmt instanceof ReturnStmt) {
						Stmt nullReturnStmt = this.getNullReturnStatements();
						units.insertAfter(nullReturnStmt, tmp);
						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertBefore(actStmts.get(i), nullReturnStmt);
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
							}
						}
						units.remove(tmp);
					}
				} catch (Exception e) {
					logger.error(e.getMessage());
					logger.error(this.injectInfo.toString());
					return false;
				}
			}
		}

		return false;
	}

	private Stmt getNullReturnStatements() {
		ReturnStmt returnStmt = Jimple.v().newReturnStmt(NullConstant.v());
		return returnStmt;
	}

	private boolean injectParameterWithNull(Body b, Local local) {
		this.injectInfo.put("VariableName", local.getName());
		logger.debug(this.formatInjectionInfo());
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
						List<Stmt> stmts = this.getNullLocalStatements(local);
						for (int i = 0; i < stmts.size(); i++) {
							if (i == 0) {
								// here is the difference between local and parameter variable
								units.insertAfter(stmts.get(i), tmp);
							} else {
								units.insertAfter(stmts.get(i), stmts.get(i - 1));
							}
						}

						List<Stmt> actStmts = this.createActivateStatement(b);
						for (int i = 0; i < actStmts.size(); i++) {
							if (i == 0) {
								units.insertAfter(actStmts.get(i), stmts.get(stmts.size() - 1));
							} else {
								units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
							}
						}
						return true;
					} catch (Exception e) {
						logger.error(e.getMessage());
						return false;
					}

				}
			}

		}
		return false;
	}

	private List<Stmt> getNullLocalStatements(Local local) {
		List<Stmt> stmts = new ArrayList<Stmt>();
		Stmt nullLocalStmt = null;
		nullLocalStmt = Jimple.v().newAssignStmt(local, NullConstant.v());
		stmts.add(nullLocalStmt);
		return stmts;
	}

	private Map<String, List<Object>> getAllVariables(Body b) {
		Map<String, List<Object>> result = new HashMap<String, List<Object>>();
		List<Local> tmpPLocals = b.getParameterLocals();
		List<Object> pLocals = new ArrayList<Object>();
		Chain<Local> tmpLocals = b.getLocals();
		List<Object> locals = new ArrayList<Object>();
		Chain<SootField> tmpFields = b.getMethod().getDeclaringClass().getFields();
		List<Object> fields = new ArrayList<Object>();

		List<Object> returnStmt = new ArrayList<Object>();

		int i;
		for (i = 0; i < tmpPLocals.size(); i++) {
			Local local = tmpPLocals.get(i);
			if (this.isTargetedType(local.getType()) && (!local.getName().startsWith("$"))) {
				pLocals.add(local);
			}
		}

		Iterator<Local> tmpLocalsItr = tmpLocals.iterator();
		while (tmpLocalsItr.hasNext()) {
			Local local = tmpLocalsItr.next();
			if (!pLocals.contains(local)) {
				if (this.isTargetedType(local.getType()) && (!local.getName().startsWith("$"))) {
					locals.add(local);
				}
			}

		}

		// actually soot cannot get Class Fields
		Iterator<SootField> tmpFieldItr = tmpFields.iterator();
		while (tmpFieldItr.hasNext()) {
			SootField field = tmpFieldItr.next();
//			logger.info("Field name:　" + field.getName() + " Field type: " + field.getType().toString());
			if (this.isTargetedType(field.getType()) && (!field.getName().startsWith("$"))) {
				if (((field.getModifiers() & Modifier.FINAL) ^ Modifier.FINAL) == 0) {// not sure
					// this is a final field, we cannot change value of it
//					logger.info(field.getDeclaration());
				} else {
					fields.add(field);
				}

			}
		}

		// for the variable used by return
		Type returnType = b.getMethod().getReturnType();
		if (!(returnType instanceof VoidType)) {
			Iterator<Unit> unitItr = b.getUnits().snapshotIterator();
			while (unitItr.hasNext()) {
				Unit unit = unitItr.next();
				if (unit instanceof ReturnStmt) {
					returnStmt.add((Stmt) unit);
				}
			}
		}

		result.put("parameter", pLocals);
		result.put("local", locals);
		result.put("field", fields);
		result.put("return", returnStmt);
		return result;
	}

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
		} else if (type instanceof VoidType) {
			return false;
		} else {
			return true;
		}
	}

	private List<Object> getQualifiedVariables(Body b, Map<String, List<Object>> allVariables, String scope) {
		// just copy list
		List<Object> variablesInScope = allVariables.get(scope);
		return new ArrayList<Object>(variablesInScope);
	}

	// we should decide whether choose return as a scope
	private List<String> getTargetScope(String variableScope) {
		List<String> scopes = new ArrayList<String>();
		if ((variableScope == null) || (variableScope == "")) {
			scopes.add("local");
			scopes.add("field");
			scopes.add("parameter");
			scopes.add("return");
		} else {
			scopes.add(variableScope);
		}
		return scopes;
	}

}
