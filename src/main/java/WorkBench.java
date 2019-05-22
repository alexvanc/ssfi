
import java.io.IOException;

public class WorkBench {
    public static void main(String[] args) throws IOException {
        WorkBench test = new WorkBench();
        long testNumber = 9;
        String testText = "test";
        test.calculate(testNumber);
        test.tryAndCatch(testText);
        test.throwException(testNumber);
    }

    public long calculate(long testNumber) {
        System.out.println("Calculation result: " + testNumber);
        return testNumber;
    }

    public void tryAndCatch(String flag) {
        try {
            if (flag.equalsIgnoreCase("test")) {
                System.out.println("Equal");
            } else {
                System.out.println("Not equal");
            }

        } catch (NullPointerException e) {
            System.out.println("Exception");
        }
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
