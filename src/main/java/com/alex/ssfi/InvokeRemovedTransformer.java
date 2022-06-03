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
import soot.jimple.InvokeStmt;
import soot.jimple.Jimple;
import soot.jimple.Stmt;
import soot.util.Chain;

public class InvokeRemovedTransformer extends BasicTransformer {
    private final Logger logger = LogManager.getLogger(InvokeRemovedTransformer.class);

    public InvokeRemovedTransformer(RunningParameter parameters) {
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
        this.injectInfo.put("FaultType", "UNUSED_INVOKE_REMOVED_FAULT");
        this.injectInfo.put("Package", targetMethod.getDeclaringClass().getPackageName());
        this.injectInfo.put("Class", targetMethod.getDeclaringClass().getName());
        this.injectInfo.put("Method", targetMethod.getSubSignature());

        logger.debug("Try to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
                + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));

        // finally perform injection
        if (this.inject(b)) {
            this.parameters.setInjected(true);
            this.recordInjectionInfo();
            logger.debug("Succeed to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
                    + this.injectInfo.get("Class") + " " + this.injectInfo.get("Method"));
            return;
        } else {
            logger.debug("Failed injection:+ " + this.formatInjectionInfo());
        }

        logger.debug("Fail to inject UNUSED_INVOKE_REMOVED_FAULT into " + this.injectInfo.get("Package") + " "
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

            if ((!withSpefcifiedMethod) && (!method.getName().contains("<init>")) && (!method.getName().contains("<clinit>"))) {
                //methods like <init> are used during verification perios, shouldn't be injected
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

        this.allQualifiedMethods = allQualifiedMethods;
    }

    private boolean inject(Body b) {

        try {
            List<Stmt> allInvokeStmts = this.getAllInvokeStmts(b);
            while (true) {
                int invokeStmtSize = allInvokeStmts.size();
                if (invokeStmtSize == 0) {
                    break;
                }
                int stmtIndex = new Random(System.currentTimeMillis()).nextInt(invokeStmtSize);
                Stmt targetInvokeStmt = allInvokeStmts.get(stmtIndex);
                allInvokeStmts.remove(stmtIndex);
                if (this.injectInvokeRemoval(b, targetInvokeStmt)) {
                    return true;
                }
            }

        } catch (Exception e) {
            logger.error(e.getMessage());
            return false;
        }
        return false;
    }

    private boolean injectInvokeRemoval(Body b, Stmt targetStmt) {
        Chain<Unit> units = b.getUnits();
        Unit nextUnit = units.getSuccOf(targetStmt);

        List<Stmt> preStmts = this.getPrecheckingStmts(b);
        for (int i = 0; i < preStmts.size(); i++) {
            if (i == 0) {
                units.insertBefore(preStmts.get(i), targetStmt);
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
        for (int i = 0; i < actStmts.size(); i++) {
            if (i == 0) {
                units.insertAfter(actStmts.get(i), conditionStmts.get(conditionStmts.size() - 1));
            } else {
                units.insertAfter(actStmts.get(i), actStmts.get(i - 1));
            }
        }

        GotoStmt skipInvoke = Jimple.v().newGotoStmt(nextUnit);
        units.insertAfter(skipInvoke, actStmts.get(actStmts.size() - 1));

        return true;
    }

    private List<Stmt> getAllInvokeStmts(Body b) {
        List<Stmt> allInvokeStmts = new ArrayList<Stmt>();
        Iterator<Unit> unitItr = b.getUnits().snapshotIterator();
        while (unitItr.hasNext()) {
            Unit tmpUnit = unitItr.next();
            if (tmpUnit instanceof InvokeStmt) {
                allInvokeStmts.add((Stmt) tmpUnit);
            }

        }
        return allInvokeStmts;
    }

}
