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

        ObjectMapper mapper = new ObjectMapper(new YAMLFactory()); // 构建YAML Mapper，用以从配置文件提取配置项
        Configuration config;
        try {
            config = mapper.readValue(new File(args[0]), Configuration.class); // 从配置文件，读取配置
            if (!config.validateConfig()) { // 验证配置项合法性
                logger.fatal("Invalid Configuration!");
                return;
            }
            if (InjectionManager.getManager(config).startInjection()) { // 根据配置，执行故障注入
                logger.info("Injection Succeed!"); // 若故障注入成功
            } else { // 若故障注入失败
                logger.error("Injection Failed!");
            }

        } catch (Exception e) { // 若注入过程发生异常
            logger.info("Injection Failed!");
            logger.fatal(e.getMessage());
            System.exit(0);
        }
    }
}
