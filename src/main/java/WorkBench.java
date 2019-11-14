import java.io.IOException;

public class WorkBench {
    public static int testNumber;
    public static boolean flag = false;

    public static void main(String[] args) throws IOException {
        WorkBench test = new WorkBench();
        WorkBench.testNumber = 9;
        String testText = "test";

        // Start of VALUE_FAULT test
        // System.out.println(test.calculate(3));
        // System.out.println(test.calculate(3));
        // System.out.println(test.calculate(3));
        // System.out.println(test.calculate(3));
        // System.out.println(test.calculate(3));
        // End of VALUE_FAULT

        // Start of SWITCH_FALLTHROUGH_FAULT test
        // test.calculate(1);
        // test.calculate(1);
        // test.calculate(1);
        // test.calculate(1);
        // test.calculate(1);
        // Start of SWITCH_MISS_DEFAULT_FAULT test
        // test.calculate(2);
        // test.calculate(2);
        // test.calculate(2);
        // test.calculate(2);
        // test.calculate(2);
        // End of SWITCH_FALLTHROUGH_FAULT + SWITCH_MISS_DEFAULT_FAULT test

        // Start of CONDITION_BORDER_FAULT + ATTRIBUTE_SHADOWED_FAULT test
//        test.calculate(5);
        test.syncMethod();
        test.syncBlock();
        // test.calculate(5);
        // test.calculate(5);
        // test.calculate(5);
        // test.calculate(5);
        // End of CONDITION_BORDER_FAULT + ATTRIBUTE_SHADOWED_FAULT test

        // Start of CONDITION_INVERSED_FAULT + UNUSED_INVOKE_REMOVED_FAULT test
        // test.calculate(4);
        // test.calculate(4);
        // test.calculate(4);
        // test.calculate(4);
        // test.calculate(4);
        // End of CONDITION_INVERSED_FAULT + UNUSED_INVOKE_REMOVED_FAULTtest

        // Start of EXCEPTION_SHORTCIRCUIT_FAULT + NULL_FAULT test
        // testText = "testEqual";
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // End of NULL_FAULT test
        // test.throwException(9);
        // test.throwException(9);
        // test.throwException(9);
        // test.throwException(9);
        // test.throwException(9);
        // End of EXCEPTION_SHORTCIRCUIT_FAULT test

        // Start of EXCEPTION_UNCAUGHT_FAULT + EXCEPTION_UNHANDLED_FAULT test
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // test.tryAndCatch(testText);
        // End of EXCEPTION_UNCAUGHT_FAULT + EXCEPTION_UNHANDLED_FAULT test

        // test.throwException(10);
        // test.testBool();
    }

//    public int calculate(int testNumber) {
//
//        // Start of CONDITION_BORDER_FAULT + CONDITION_INVERSED_FAULT test
//        // if(testNumber<5) {
//        // System.out.println("Number smaller than 5");
//        // }else {
//        // System.out.println("Number larger/euqal than 5");
//        // }
//        // End of CONDITION_BORDER_FAULT + CONDITION_INVERSED_FAULT test
//
//        // Start of UNUSED_INVOKE_REMOVED_FAULT test
//        // System.out.println("For removel");
//        // System.out.println("For second removel\n");
//        // End of UNUSED_INVOKE_REMOVED_FAULT test
//
//        // Start of ATTRIBUTE_SHADOWED_FAULT test
//        System.out.println("Field" + this.testNumber);
//        System.out.println("Local" + testNumber + "\n");
//        // End of ATTRIBUTE_SHADOWED_FAULT test
//
//        // for(int i=0;i<5;i++) {
//        // this.testNumber=this.testNumber+1;
//        // // }
//        // System.out.println("Calculation result: " + testNumber);
//        // return testNumber;
//        // return this.testNumber;
//
//        // Start of SWITCH_FALLTHROUGH_FAULT + SWITCH_MISS_DEFAULT_FAULTtest
//        // switch (testNumber) {
//        // case 1:
//        // this.testNumber = 2;
//        // break;
//        // case 2:
//        // this.testNumber = testNumber + 3;
//        // case 3:
//        // this.testNumber = 4;
//        // case 4:
//        // this.testNumber = 5;
//        // default:
//        // this.testNumber = 6;
//        // }
//        // System.out.println(this.testNumber);
//        // End of SWITCH_FALLTHROUGH_FAULT + SWITCH_MISS_DEFAULT_FAULT test
//
//        // Start of VALUE_FAULT test
//        // int test=testNumber+this.testNumber;
//        // End of VALUE_FAULT test
//
//        return this.testNumber;
//    }

//    public void tryAndCatch(String flag) {
//
//        // Start of EXCEPTION_UNCAUGHT_FAULT + EXCEPTION_UNHANDLED_FAULT +
//        // NULL_FAULT test
//        try {
//            if (flag.equalsIgnoreCase("test")) {
//                System.out.println("Equal");
//                throw new NullPointerException("test");
//            } else {
//                System.out.println("Not equal");
//
//            }
//
//        } catch (NullPointerException e) {
//            System.out.println("Exception");
//
//        }
//        // End of EXCEPTION_UNCAUGHT_FAULT + EXCEPTION_UNHANDLED_FAULT +
//        // NULL_FAULT test
//
//        // catch (Exception e) {
//        // System.out.println("Other Exception");
//        // }
//        // End of NULL_FAULT test
//
//        // System.out.println();
//        // System.out.println("OutCatch");
//        return;
//
//    }

//    public void throwException(long limit) throws IOException {
//
//        // Start of CONDITION_BORDER_FAULT + CONDITION_INVERSED_FAULT test
//        if (limit > 10) {
//            throw new IOException();
//        } else {
//            System.out.println("Smaller than 10, no exeception");
//            return;
//        }
//
//    }

    public synchronized void syncMethod() {
        int result = 2;
        System.out.println("Result is: " + result);
    }

    public void syncBlock() {
        int result = 3;
        synchronized (this) {
            System.out.println("Result is " + result);
        }
    }

//    public void testBool() {
//        if (WorkBench.flag) {
//            System.out.println("True");
//        } else {
//            System.out.println("False");
//        }
//    }

}

class TestInnerClass {
    private int test = 0;
}
