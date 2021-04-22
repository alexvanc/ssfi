package com.alex.ssfi;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import soot.tagkit.AttributeValueException;
import soot.tagkit.Tag;

public class RandomTag implements Tag {
	private long targetCounter;
	
	public RandomTag(int targetCounter) {
		this.targetCounter=targetCounter;
	}

	public String getName() {
		// TODO Auto-generated method stub
		return "sootTargetCounter";
	}

	public byte[] getValue() throws AttributeValueException {
		// TODO Auto-generated method stub
		ByteArrayOutputStream baos = new ByteArrayOutputStream(4);
        DataOutputStream dos = new DataOutputStream(baos);
        try {
            dos.writeLong(targetCounter);
            dos.flush();
        } catch(IOException e) {
            System.err.println(e);
            throw new RuntimeException(e);
        }
        return baos.toByteArray();
	}

	public long getTargetCounter() {
		return targetCounter;
	}


}
