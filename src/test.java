public class test {
    public static void main(String[] args) {
        int value=121;
        double []binary=new double[32];
        int index=31;
        do{
            binary[index--]=(value&1);
            value >>>=1;
            System.out.println(value);
        }while (value!=0);

        for (int i = 0; i <32 ; i++) {
            System.out.println("b"+i+"  "+binary[i]);
        }
    }
}
