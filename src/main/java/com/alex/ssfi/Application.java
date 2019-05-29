package com.alex.ssfi;


import com.fasterxml.jackson.databind.ObjectMapper;


public class Application {
    public static void main( String[] args )
    {
        if (args.length<=2){
            return;
        }
//        Configuration config=new Con
        ObjectMapper mapper=new ObjectMapper(args[1]);

    }
}
