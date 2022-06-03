package com.alex.ssfi.util;

import java.util.Random;

import soot.BooleanType;
import soot.ByteType;
import soot.CharType;
import soot.DoubleType;
import soot.FloatType;
import soot.IntType;
import soot.LongType;
import soot.ShortType;
import soot.Type;

public class FaultTypeHelper {
    public static final int BYTE = 0;
    public static final int CHAR = 1;
    public static final int INT = 2;
    public static final int LONG = 3;
    public static final int SHORT = 4;
    public static final int BOOL = 5;
    public static final int FLOAT = 6;
    public static final int DOUBLE = 7;
    public static final int STRING = 8;


    public static String getRandomValueType() {
        Random rand = new Random();
        int result = rand.nextInt(STRING + 1);
        return getValueNameByTypeID(result);

    }

    public static boolean isTheSameType(Type source, String target) {
        int valueTypeID = getValueTypeIDByName(target);

        switch (valueTypeID) {
            case BYTE:
                return source instanceof ByteType;
            case CHAR:
                return source instanceof CharType;
            case INT:
                return source instanceof IntType;
            case LONG:
                return source instanceof LongType;
            case SHORT:
                return source instanceof ShortType;
            case BOOL:
                return source instanceof BooleanType;
            case FLOAT:
                return source instanceof FloatType;
            case DOUBLE:
                return source instanceof DoubleType;
            case STRING:
                return source.toString().equals("java.lang.String");
            default:
                return false;
        }
    }

    public static int getValueTypeIDByName(String name) {
        if (name == null) {
            return -1;
        }
        String lowName = name.toUpperCase();
        if (lowName.equals("BYTE")) {
            return BYTE;
        } else if (lowName.equals("CHAR")) {
            return CHAR;
        } else if (lowName.equals("INT")) {
            return INT;
        } else if (lowName.equals("LONG")) {
            return LONG;
        } else if (lowName.equals("SHORT")) {
            return SHORT;
        } else if (lowName.equals("BOOL")) {
            return BOOL;
        } else if (lowName.equals("FLOAT")) {
            return FLOAT;
        } else if (lowName.equals("DOUBLE")) {
            return DOUBLE;
        } else if (lowName.equals("STRING")) {
            return STRING;
        } else {
            return -1;
        }
    }

    public static String getValueNameByTypeID(int valueID) {
        switch (valueID) {
            case BYTE:
                return "BYTE";
            case CHAR:
                return "CHAR";
            case INT:
                return "INT";
            case LONG:
                return "LONG";
            case SHORT:
                return "SHORT";
            case BOOL:
                return "BOOL";
            case FLOAT:
                return "FLOAT";
            case DOUBLE:
                return "DOUBLE";
            case STRING:
                return "STRING";
            default:
                return null;
        }
    }

}
