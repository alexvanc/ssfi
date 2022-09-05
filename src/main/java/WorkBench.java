import java.io.IOException;

public class WorkBench {
    public int testNumber = 9;

    public static void main(String[] args) throws IOException {
        WorkBench test = new WorkBench();
        String testText = "test";
        String[] testPointer = {"hello", "world"};;
        int inputNumber = 2;
        test.assign(inputNumber);
        test.calculate(inputNumber);
        test.checkPointer(testPointer);
        test.tryAndCatch(testText);
        test.throwException(test.testNumber);
    }
    public long assign(int testNumber){
         switch (testNumber) {
         case 1:
         	this.testNumber = 2;
         	break;
         case 2:
         	this.testNumber = 3;
             break;
         case 3:
         	this.testNumber = 4;
             break;
         default:
             this.testNumber = 5;
         }
        System.out.println("Input number: " + testNumber);
        System.out.println("Assigned result: " + this.testNumber);
         return this.testNumber;
    }

    public void checkPointer(String[] pointer){
        if (pointer==null){
            System.out.println("pointer is null");
        }
    }

    public long calculate(int testNumber) {
        for (int i = 0; i < testNumber; i++) {
            this.testNumber = this.testNumber + 1;
        }
        System.out.println("Calculation result: " + this.testNumber);
        return this.testNumber;
    }

    public void tryAndCatch(String flag) {
        try {
            if (flag.equalsIgnoreCase("test")) {
                System.out.println("Equal");
                throw new IOException();
            } else {
                System.out.println("Not equal");
            }
        } catch (Exception e) {
            System.out.println("OutCatch");
        }
    }

    public void throwException(long limit) throws IOException {
        if (limit > 5) {
            throw new IOException();
        } else {
            System.out.println("Smaller than 5, no exception");
        }
    }
}

//class TestInnerClass {
//    private final int test = 0;
//}