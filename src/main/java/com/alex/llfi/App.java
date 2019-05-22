package com.alex.llfi;

import soot.Main;
import soot.PackManager;
import soot.Transform;

/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {
        PackManager.v().getPack("jtp").add(new Transform("jtp.instrumenter", new CatchTransformer()) );
        Main.main(args);
    }
}
