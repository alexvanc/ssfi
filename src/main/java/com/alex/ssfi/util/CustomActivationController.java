package com.alex.ssfi.util;

public class CustomActivationController {

    public static boolean shouldActivate(String faultID, String unit, int lineNumber) {
        //users define their own logic here
        //by default, it equals to the ALWAYS activation mode
        return true;
    }

    public static boolean activate(String faultID){
//        users record activation info
        return true;
    }
    
}
