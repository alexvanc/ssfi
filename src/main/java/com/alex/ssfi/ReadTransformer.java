package com.alex.ssfi;

import java.util.Iterator;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.Reporter;
import com.alex.ssfi.util.RunningParameter;

import soot.Body;
import soot.BodyTransformer;
import soot.Unit;
import soot.jimple.Stmt;
import soot.util.Chain;

/*
 * this class is only for learning how it identify different Jimple Statements
 */
public class ReadTransformer extends BodyTransformer {
    private final RunningParameter parameters;
    private static final Logger logger = LogManager.getLogger(ValueTransformer.class);

    public ReadTransformer(RunningParameter parameters) {
        this.parameters = parameters;

    }

    @Override
    protected void internalTransform(Body b, String phaseName, Map<String, String> options) {
        // TODO Auto-generated method stub
        String methodSignature = b.getMethod().getSignature();
        logger.info("Method signature:" + methodSignature);
        logger.info("Phase Name:" + phaseName);
        logger.info("Start to process Units:");
        Chain<Unit> units = b.getUnits();
        Iterator<Unit> stmIt = units.snapshotIterator();
        while (stmIt.hasNext()) {
            Stmt stmp = (Stmt) stmIt.next();
            Reporter.checkStmtType(stmp);
        }
        logger.info("Done to process Units\n");

    }

}
