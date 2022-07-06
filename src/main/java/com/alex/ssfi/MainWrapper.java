package com.alex.ssfi;


import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Date;

import soot.CompilationDeathException;
import soot.G;
import soot.Main;
import soot.OptionsParseException;
import soot.Pack;
import soot.PackManager;
import soot.PhaseOptions;
import soot.Scene;
import soot.Singletons;
import soot.Timers;
import soot.Transform;
import soot.options.CGOptions;
import soot.options.Options;
import soot.toolkits.astmetrics.ClassData;

/**
 * Main class for Soot; provides Soot's command-line user interface.
 */
public class MainWrapper {
    public MainWrapper(Singletons.Global g) {
    }

    public static Main v() {
        return G.v().soot_Main();
    }

    public static final String versionString = MainWrapper.class.getPackage().getImplementationVersion() == null ? "trunk"
            : MainWrapper.class.getPackage().getImplementationVersion();

    private Date start;
    private long startNano;
    private long finishNano;

    private void printVersion() {
        System.out.println("Soot version " + versionString);
        System.out.println("Copyright (C) 1997-2010 Raja Vallee-Rai and others.");
        System.out.println("All rights reserved.");
        System.out.println();
        System.out.println("Contributions are copyright (C) 1997-2010 by their respective contributors.");
        System.out.println("See the file 'credits' for a list of contributors.");
        System.out.println("See individual source files for details.");
        System.out.println();
        System.out.println("Soot comes with ABSOLUTELY NO WARRANTY.  Soot is free software,");
        System.out.println("and you are welcome to redistribute it under certain conditions.");
        System.out.println("See the accompanying file 'COPYING-LESSER.txt' for details.");

        System.out.println("Visit the Soot website:");
        System.out.println("  http://www.sable.mcgill.ca/soot/");

        System.out.println("For a list of command line options, enter:");
        System.out.println("  java soot.Main --help");
    }

    private void processCmdLine(String[] args) {

        if (!Options.v().parse(args)) {
            throw new OptionsParseException("Option parse error");
        }

        if (PackManager.v().onlyStandardPacks()) {
            for (Pack pack : PackManager.v().allPacks()) {
                Options.v().warnForeignPhase(pack.getPhaseName());
                for (Transform tr : pack) {
                    Options.v().warnForeignPhase(tr.getPhaseName());
                }
            }
        }
        Options.v().warnNonexistentPhase();

        if (Options.v().help()) {
            System.out.println(Options.v().getUsage());
            throw new CompilationDeathException(CompilationDeathException.COMPILATION_SUCCEEDED);
        }

        if (Options.v().phase_list()) {
            System.out.println(Options.v().getPhaseList());
            throw new CompilationDeathException(CompilationDeathException.COMPILATION_SUCCEEDED);
        }

        if (!Options.v().phase_help().isEmpty()) {
            for (String phase : Options.v().phase_help()) {
                System.out.println(Options.v().getPhaseHelp(phase));
            }
            throw new CompilationDeathException(CompilationDeathException.COMPILATION_SUCCEEDED);
        }

        if ((!Options.v().unfriendly_mode() && (args.length == 0)) || Options.v().version()) {
            printVersion();
            throw new CompilationDeathException(CompilationDeathException.COMPILATION_SUCCEEDED);
        }

        if (Options.v().on_the_fly()) {
            Options.v().set_whole_program(true);
            PhaseOptions.v().setPhaseOption("cg", "off");
        }

        postCmdLineCheck();
    }

    private void postCmdLineCheck() {
        if (Options.v().classes().isEmpty() && Options.v().process_dir().isEmpty()) {
            throw new CompilationDeathException(CompilationDeathException.COMPILATION_ABORTED, "No input classes specified!");
        }
    }

    public String[] cmdLineArgs = new String[0];

    /**
     * Entry point for cmd line invocation of soot.
     */
    public static void main(String[] args) {
        try {
            Main.v().run(args);
        } catch (OptionsParseException e) {
            // error message has already been printed
        } catch (StackOverflowError e) {
            System.err.println("" + "Soot has run out of stack memory.");
            System.err.println("" + "To allocate more stack memory to Soot, use the -Xss switch to Java.");
            System.err.println("" + "For example (for 2MB): java -Xss2m soot.Main ...");
            throw e;
        } catch (OutOfMemoryError e) {
            System.err.println("" + "Soot has run out of the memory allocated to it by the Java VM.");
            System.err.println("" + "To allocate more memory to Soot, use the -Xmx switch to Java.");
            System.err.println("" + "For example (for 2GB): java -Xmx2g soot.Main ...");
            throw e;
        }
    }

    /**
     * Entry point to the soot's compilation process.
     */
    public void run(String[] args) {
        cmdLineArgs = args;

        start = new Date();
        startNano = System.nanoTime();

        try {
            Timers.v().totalTimer.start();

            processCmdLine(cmdLineArgs);

            autoSetOptions();

            System.out.println("Soot started on " + start);

            Scene.v().loadNecessaryClasses();

            /*
             * By this all the java to jimple has occured so we just check ast-metrics flag
             *
             * If it is set......print the astMetrics.xml file and stop executing soot
             */
            if (Options.v().ast_metrics()) {
                try {
                    OutputStream streamOut = new FileOutputStream("../astMetrics.xml");
                    PrintWriter writerOut = new PrintWriter(new OutputStreamWriter(streamOut));
                    writerOut.println("<?xml version='1.0'?>");
                    writerOut.println("<ASTMetrics>");

                    for (ClassData cData : G.v().ASTMetricsData) {
                        // each is a classData object
                        writerOut.println(cData);
                    }

                    writerOut.println("</ASTMetrics>");
                    writerOut.flush();
                    streamOut.close();
                } catch (IOException e) {
                    throw new CompilationDeathException("Cannot output file astMetrics", e);
                }
                return;
            }

            PackManager.v().runPacks();
            if (!Options.v().oaat()) {
                PackManager.v().writeOutput();
            }

            Timers.v().totalTimer.end();

            // Print out time stats.
            if (Options.v().time()) {
                Timers.v().printProfilingInformation();
            }

        } catch (CompilationDeathException e) {
            Timers.v().totalTimer.end();
            if (e.getStatus() != CompilationDeathException.COMPILATION_SUCCEEDED) {
                throw e;
            } else {
                return;
            }
        }

        finishNano = System.nanoTime();

        System.out.println("Soot finished on " + new Date());
        long runtime = (finishNano - startNano) / 1000000l;
        System.out.println("" + "Soot has run for " + (runtime / 60000) + " min. " + ((runtime % 60000) / 1000) + " sec.");

    }

    public void autoSetOptions() {
        // when no-bodies-for-excluded is enabled, also enable phantom refs
        if (Options.v().no_bodies_for_excluded()) {
            Options.v().set_allow_phantom_refs(true);
        }

        // when reflection log is enabled, also enable phantom refs
        CGOptions cgOptions = new CGOptions(PhaseOptions.v().getPhaseOptions("cg"));
        String log = cgOptions.reflection_log();
        if ((log != null) && (log.length() > 0)) {
            Options.v().set_allow_phantom_refs(true);
        }

        // if phantom refs enabled, ignore wrong staticness in type assigner
        if (Options.v().allow_phantom_refs()) {
            Options.v().set_wrong_staticness(Options.wrong_staticness_fix);
        }
    }
}
