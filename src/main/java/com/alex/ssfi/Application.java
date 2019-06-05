package com.alex.ssfi;



import java.io.File;

import com.alex.ssfi.util.Configuration;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;


public class Application {
    public static void main( String[] args )
    {
        if (args.length<=0){
            return;
        }
        ObjectMapper mapper=new ObjectMapper(new YAMLFactory());
        Configuration config=null;
        try {
        	config=mapper.readValue(new File(args[0]), Configuration.class);
        	if(!config.validateConfig()) {
        		System.out.println("Invalid Configuration!");
        		return;
        	}
        	InjectionManager.getManager().peformInjection(config);
        	System.out.println("Success");
        }catch (Exception e){
        	System.out.println(e.getMessage());
        	System.exit(0);
        }

    }
}
