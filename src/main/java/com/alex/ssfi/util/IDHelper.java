package com.alex.ssfi.util;

import org.apache.commons.lang3.RandomStringUtils;

public class IDHelper {
	public static String generateID(int length) {
		boolean userLetters=true;
		boolean useNumbers=false;
		String result =RandomStringUtils.random(length,userLetters,useNumbers);
		return result;
	}

}
