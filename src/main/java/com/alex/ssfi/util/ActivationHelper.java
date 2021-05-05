package com.alex.ssfi.util;

public class ActivationHelper {

    public static boolean hasActivated(String faultID){
        //users define their own logic here
        //by default, it equals to the ALWAYS activation mode
        return false;
    }

    public static boolean activate(String faultID){
        return true;
    }
    
}
