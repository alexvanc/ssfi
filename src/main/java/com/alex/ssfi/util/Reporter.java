package com.alex.ssfi.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import soot.jimple.AssignStmt;
import soot.jimple.EnterMonitorStmt;
import soot.jimple.ExitMonitorStmt;
import soot.jimple.GotoStmt;
import soot.jimple.IdentityStmt;
import soot.jimple.IfStmt;
import soot.jimple.InvokeStmt;
import soot.jimple.LookupSwitchStmt;
import soot.jimple.NopStmt;
import soot.jimple.ReturnStmt;
import soot.jimple.ReturnVoidStmt;
import soot.jimple.Stmt;
import soot.jimple.TableSwitchStmt;
import soot.jimple.ThrowStmt;

public class Reporter {
    private static final Logger logger = LogManager.getLogger(Reporter.class);

    public static JStatementEnum checkStmtType(Stmt stmt) {

        if (stmt instanceof IdentityStmt) {
            logger.info("In Identity: " + stmt);
            return JStatementEnum.IDENTITY;
        } else if (stmt instanceof AssignStmt) {
            logger.info("In Assign: " + stmt);
            return JStatementEnum.ASSIGN;
        } else if (stmt instanceof NopStmt) {
            logger.info("In Nop: " + stmt);
            return JStatementEnum.NOP;
        } else if (stmt instanceof ThrowStmt) {
            logger.info("In throw: " + stmt);
            return JStatementEnum.THROW;
        } else if (stmt instanceof InvokeStmt) {
            logger.info("In invoke: " + stmt);
            return JStatementEnum.INVOKE;
        } else if (stmt instanceof GotoStmt) {
            logger.info("In Goto: " + stmt);
            return JStatementEnum.GOTO;
        } else if (stmt instanceof IfStmt) {
            logger.info("In if: " + stmt);
            return JStatementEnum.IF;
        } else if ((stmt instanceof ReturnStmt)) {
            logger.info("In return: " + stmt);
            return JStatementEnum.RETURN;
        } else if (stmt instanceof ReturnVoidStmt) {
            logger.info("In return void: " + stmt);
            return JStatementEnum.RETURNVOID;
        } else if (stmt instanceof TableSwitchStmt) {
            logger.info("In table switch: " + stmt);
            return JStatementEnum.TABLESWITCH;
        } else if (stmt instanceof LookupSwitchStmt) {
            logger.info("In lookup switch: " + stmt);
            return JStatementEnum.LOOKUPSWITCH;
        } else if (stmt instanceof EnterMonitorStmt) {
            logger.info("In enter monitoring: " + stmt);
            return JStatementEnum.ENTERMONITOR;
        } else if (stmt instanceof ExitMonitorStmt) {
            logger.info("In exit monitoring: " + stmt);
            return JStatementEnum.RETURNVOID;
        } else {
            logger.info("In Other");
            return JStatementEnum.OTHER;
        }
    }
}
