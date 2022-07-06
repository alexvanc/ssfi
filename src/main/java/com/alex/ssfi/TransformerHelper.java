package com.alex.ssfi;

import com.alex.ssfi.util.RunningParameter;

import soot.BodyTransformer;

import java.util.Random;

public class TransformerHelper {
    public static final int VALUE_FAULT = 0; // VALUE_FAULT对应的ID为0
    public static final int NULL_FAULT = 1; // NULL_FAULT对应的ID为1
    public static final int EXCEPTION_SHORTCIRCUIT_FAULT = 2; // EXCEPTION_SHORTCIRCUIT_FAULT对应的ID为2
    public static final int EXCEPTION_UNCAUGHT_FAULT = 3; // EXCEPTION_UNCAUGHT_FAULT对应的ID为3
    public static final int EXCEPTION_UNHANDLED_FAULT = 4; // EXCEPTION_UNHANDLED_FAULT对应的ID为4
    public static final int ATTRIBUTE_SHADOWED_FAULT = 5; // ATTRIBUTE_SHADOWED_FAULT对应的ID为5
    public static final int SWITCH_FALLTHROUGH_FAULT = 6; // SWITCH_FALLTHROUGH_FAULT对应的ID为6
    public static final int SWITCH_MISS_DEFAULT_FAULT = 7; // SWITCH_MISS_DEFAULT_FAULT对应的ID为7
    public static final int UNUSED_INVOKE_REMOVED_FAULT = 8; // UNUSED_INVOKE_REMOVED_FAULT对应的ID为8
    public static final int CONDITION_BORDER_FAULT = 9; // CONDITION_BORDER_FAULT对应的ID为9
    public static final int CONDITION_INVERSED_FAULT = 10; // CONDITION_INVERSED_FAULT对应的ID为10
    public static final int SYNC_FAULT = 11; // SYNC_FAULT对应的ID为11

    // 根据运行时参数和故障类型的名称，获取对应的BodyTransformer
    public static BodyTransformer getTransformer(String faultType, RunningParameter parameter) {
        int faultTypeID = getFaultTypeIDByName(faultType); //根据故障类型的名字获取其ID
        switch (faultTypeID) { // 根据故障类型的ID，获取对应的BodyTransformer
            case VALUE_FAULT:
                return new ValueTransformer(parameter);
            case NULL_FAULT:
                return new NullTransformer(parameter);
            case EXCEPTION_SHORTCIRCUIT_FAULT:
                return new ExceptionShortCircuitTransformer(parameter);
            case EXCEPTION_UNCAUGHT_FAULT:
                return new ExceptionUncaughtTransformer(parameter);
            case EXCEPTION_UNHANDLED_FAULT:
                return new ExceptionUnHandledTransformer(parameter);
            case SWITCH_FALLTHROUGH_FAULT:
                return new SwitchFallThroughTransformer(parameter);
            case SWITCH_MISS_DEFAULT_FAULT:
                return new SwitchMissDefaultTransformer(parameter);
            case UNUSED_INVOKE_REMOVED_FAULT:
                return new InvokeRemovedTransformer(parameter);
            case CONDITION_BORDER_FAULT:
                return new ConditionBorderTransformer(parameter);
            case CONDITION_INVERSED_FAULT:
                return new ConditionInversedTransformer(parameter);
            case SYNC_FAULT:
                return new SyncTransformer(parameter);
            default: // 若指定的故障为其他类型，返回的Transformer不对代码做任何操作
                return new ReadTransformer(parameter);
        }
    }

    //根据故障类型的名字获取其ID
    public static int getFaultTypeIDByName(String faultType) {
        if ((faultType == null) || (faultType.equals(""))) {
//			if fault type is not specified, this configuration item will be randomly assigned a value
            return new Random().nextInt(12);
        }
        String upperCaseName = faultType.toUpperCase();  // 将故障类型的名字转换为全大写
        if (upperCaseName.equals("VALUE_FAULT")) {
            return VALUE_FAULT;
        } else if (upperCaseName.equals("NULL_FAULT")) {
            return NULL_FAULT;
        } else if (upperCaseName.equals("EXCEPTION_SHORTCIRCUIT_FAULT")) {
            return EXCEPTION_SHORTCIRCUIT_FAULT;
        } else if (upperCaseName.equals("EXCEPTION_UNCAUGHT_FAULT")) {
            return EXCEPTION_UNCAUGHT_FAULT;
        } else if (upperCaseName.equals("EXCEPTION_UNHANDLED_FAULT")) {
            return EXCEPTION_UNHANDLED_FAULT;
        } else if (upperCaseName.equals("ATTRIBUTE_SHADOWED_FAULT")) {
            return ATTRIBUTE_SHADOWED_FAULT;
        } else if (upperCaseName.equals("SWITCH_FALLTHROUGH_FAULT")) {
            return SWITCH_FALLTHROUGH_FAULT;
        } else if (upperCaseName.equals("SWITCH_MISS_DEFAULT_FAULT")) {
            return SWITCH_MISS_DEFAULT_FAULT;
        } else if (upperCaseName.equals("UNUSED_INVOKE_REMOVED_FAULT")) {
            return UNUSED_INVOKE_REMOVED_FAULT;
        } else if (upperCaseName.equals("CONDITION_BORDER_FAULT")) {
            return CONDITION_BORDER_FAULT;
        } else if (upperCaseName.equals("CONDITION_INVERSED_FAULT")) {
            return CONDITION_INVERSED_FAULT;
        } else if (upperCaseName.equals("SYNC_FAULT")) {
            return SYNC_FAULT;
        } else { // 若故障类型的名字不在上述类型，则返回-1，表示获取失败
            return -1;
        }
    }

}
