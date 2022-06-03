package com.alex.ssfi;


import java.io.File;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.alex.ssfi.util.Configuration;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;


public class Application {
    private static final Logger logger = LogManager.getLogger(Application.class);

    public static void main(String[] args) {
        if (args.length <= 0) {
            logger.fatal("Configuration file missing");
            return;
        }
        ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        Configuration config = null;
        try {
            config = mapper.readValue(new File(args[0]), Configuration.class);
            if (!config.validateConfig()) {
                logger.fatal("Invalid Configuration!");
                return;
            }
            if (InjectionManager.getManager().peformInjection(config)) {
                logger.info("Injection Succeed!");
            } else {
                logger.error("Injection Failed!");
            }

        } catch (Exception e) {
            logger.info("Injection Failed!");
            logger.fatal(e.getMessage());
            System.exit(0);
        }

    }
}
