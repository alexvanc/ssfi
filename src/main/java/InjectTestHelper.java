import com.alex.ssfi.InjectionManager;
import com.alex.ssfi.util.Configuration;
import com.alex.ssfi.util.LocationPattern;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;

public class InjectTestHelper {
    private static final Logger logger = LogManager.getLogger(InjectTestHelper.class);

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

            int faultType = 12;
            String MethodP = "assign";
            switch (faultType) {
                case 1:
                    config.setType("VALUE_FAULT");
                    config.setVariableScope("parameter");
                    config.setVariableType("int");
                    config.setAction("TO");
                    config.setTargetValue("1");
                    MethodP = "assign";
                    break;
                case 2:
                    config.setType("NULL_FAULT");
                    config.setVariableScope("parameter");
                    MethodP = "checkPointer";
                    break;
                case 3:
                    config.setType("EXCEPTION_SHORTCIRCUIT_FAULT");
                    config.setVariableScope("throw");
                    MethodP = "throwException";
                    break;
                case 4:
                    config.setType("EXCEPTION_UNCAUGHT_FAULT");
                    MethodP = "throwException";
                    break;
                case 5:
                    config.setType("EXCEPTION_UNHANDLED_FAULT");
                    MethodP = "tryAndCatch";
                    break;
                case 6:
                    config.setType("ATTRIBUTE_SHADOWED_FAULT");
                    config.setAction("local2field");
                    MethodP = "assign";
                    break;
                case 7:
                    config.setType("SWITCH_FALLTHROUGH_FAULT");
                    config.setAction("fallthrough");
                    MethodP = "assign";
                    break;
                case 8:
                    config.setType("SWITCH_MISS_DEFAULT_FAULT");
                    MethodP = "assign";
                    break;
                case 9:
                    config.setType("UNUSED_INVOKE_REMOVED_FAULT");
                    MethodP = "assign";
                    break;
                case 10:
                    config.setType("CONDITION_BORDER_FAULT");
                    MethodP = "throwException";
                    break;
                case 11:
                    config.setType("CONDITION_INVERSED_FAULT");
                    MethodP = "throwException";
                    break;
                case 12:
                    config.setType("SYNC_FAULT");
                    config.setVariableScope("method");
                    config.setAction("SYNC");
                    MethodP = "main";
                    break;

            }
            LocationPattern tempLocationPattern = config.getLocationPattern();
            tempLocationPattern.setMethodP(MethodP);
            config.setLocationPattern(tempLocationPattern);

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


//            config.setType("VALUE_FAULT");
//            config.setVariableScope("parameter");
//            config.setVariableType("int");
//            config.setAction("TO");
//            config.setTargetValue("1");
//            String MethodP = "assign";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("NULL_FAULT");
//            config.setVariableScope("parameter");
//            String MethodP = "checkPointer";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("EXCEPTION_SHORTCIRCUIT_FAULT");
//            config.setVariableScope("throw");
//            String MethodP = "throwException";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("EXCEPTION_UNCAUGHT_FAULT");
//            String MethodP = "assign";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("EXCEPTION_UNHANDLED_FAULT");
//            String MethodP = "tryAndCatch";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("ATTRIBUTE_SHADOWED_FAULT");
//            config.setAction("local2field");
//            String MethodP = "assign";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("SWITCH_FALLTHROUGH_FAULT");
//            config.setAction("fallthrough");
//            String MethodP = "assign";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("SWITCH_MISS_DEFAULT_FAULT");
//            String MethodP = "assign";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("UNUSED_INVOKE_REMOVED_FAULT");
//            String MethodP = "main";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("CONDITION_BORDER_FAULT");
//            String MethodP = "throwException";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("CONDITION_INVERSED_FAULT");
//            String MethodP = "throwException";
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);

//            config.setType("SYNC_FAULT");
//            String MethodP = "main";
//            config.setVariableScope("method");
//            config.setAction("SYNC");
//            LocationPattern tempLocationPattern = config.getLocationPattern();
//            tempLocationPattern.setMethodP(MethodP);
//            config.setLocationPattern(tempLocationPattern);