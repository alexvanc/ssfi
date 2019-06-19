
import java.io.IOException;

public class WorkBench {
	public long testNumber = 9;

	public static void main(String[] args) throws IOException {
		WorkBench test = new WorkBench();
		String testText = "test";
		test.calculate(2);
		test.tryAndCatch(testText);
		test.throwException(test.testNumber);
	}

	public long calculate(int testNumber) {
//    	for(int i=0;i<5;i++) {
//    		this.testNumber=this.testNumber+1;
////    	}
//        System.out.println("Calculation result: " + testNumber);
////        return testNumber;
//    		return this.testNumber;
		switch (testNumber) {
		case 1:
			this.testNumber = 2;
			break;
		case 2:
			this.testNumber = 3;
		case 3:
			this.testNumber = 4;
		}
		return this.testNumber;
	}

	public void tryAndCatch(String flag) {
		try {
			if (flag.equalsIgnoreCase("test")) {
				System.out.println("Equal");
			} else {
				System.out.println("Not equal");
			}

//        } catch (NullPointerException e) {
//            System.out.println("Exception");
		} catch (Exception e) {
			System.out.println("Other Exception");
		}
		System.out.println("OutCatch");
		return;

	}

	public void throwException(long limit) throws IOException {
		if (limit > 10) {
			throw new IOException();
		} else {
			System.out.println("Smaller than 10, no exeception");
			return;
		}
	}

}

class TestInnerClass {
	private int test = 0;
}
