package com.alex.ssfi;

import soot.Main;
import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.Transform;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {
//    	Scene.v().addBasicClass("java.io.FileWriter",SootClass.SIGNATURES);
//    	Scene.v().loadClassAndSupport("java.io.FileWriter");
//    	PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", new ValueTransformer()) );
//        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", new CatchTransformer()) );
//    	PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", new ThrowTransformer()) );
    	PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", GotoTransformer.v()) );
        Main.main(args);
    }
}
