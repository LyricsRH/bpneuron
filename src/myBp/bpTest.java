package myBp;

import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

public class bpTest {

    //输入层数组，隐含层输入，长度：输入节点个数+1
    private double[] hide_x;
    //隐含层权值  [隐含层节点个数][输入层节点个数+1]  每一行是一个隐含层节点
    private double[][] hide_w;
    //隐含层误差 长度：隐含层节点个数
    private double [] hide_error;
    //输出层输入，输出层输出 长度：隐含层节点个数+1，第一个是1
    private double[] out_x;
    //输出层权值  [输出层节点个数][隐含层节点个数+1]
    private double[][] out_w;
    //输出层误差 长度：输出层节点个数
    private double[] out_error;
    //目标值
    private double[] target;
    //学习速率
    private double rate;

    //构造函数（输入节点个数，隐含层节点个数，输出层节点个数，rate）
    public bpTest(int input,int hide,int out ,double rate){
        hide_x=new double[input+1];
        hide_w=new double[hide][input+1];
        hide_error=new double[hide];
        out_x=new double[hide+1];
        out_w=new double[out][hide+1];
        out_error=new double[out];
        target=new double[out];
        this.rate=rate;
        init_weit();
    }

    //1，1初始化权值矩阵，对数组赋初值,都为0
       public void init_weit(){
          //hide_w和out_w都要初始化
          setWeight(hide_w);
          setWeight(out_w);
       }

       public void setWeight(double [][] weight){
           for (int i = 0; i <weight.length ; i++) {
               for (int j = 0; j <weight[0].length ; j++) {
                   weight[i][j]=0;
               }
           }
       }


    //1.1设置原始输入矩阵,传入数组，
    public void setHide_x(double[]data){
        if(data.length!=hide_x.length-1){
            throw new IllegalArgumentException("数据大小与节点不匹配");
        }
        System.arraycopy(data,0,hide_x,1,data.length);
        hide_x[0]=1.0;
    }

    //1.1设置目标
    public void  setTarget(double[] target){
        this.target=target;
    }

    //2 训练 向前，向后

    public void train(double[] trainData,double[] target){
        setHide_x(trainData);
        setTarget(target);

        //向前传播
        double[] output=new double[out_w.length+1];
        forward(hide_x,output);

        backprocess(output);
    }

    //2.1反向过程  求误差----更新权值矩阵
    public void backprocess(double [] output){
        get_out_error(output,target,out_error);
        get_hide_error(out_error,out_w,hide_error,out_x);
        update_Weight(hide_w,hide_error,hide_x);
        update_Weight(out_w,out_error,out_x);
    }

    //求每层误差
    public  void get_out_error(double []output,double [] target ,double []out_error){
        for (int i = 0; i <out_error.length ; i++) {
            out_error[i]=(target[i]-output[i+1])*output[i+1]*(1d-output[i+1]);
        }
    }

    public void get_hide_error(double[]out_error,double [][]out_w,double[] hide_error,double[]out_x){
        for (int i = 0; i <hide_error.length ; i++) {
            double sum=0;
            for (int j = 0; j <out_w.length ; j++) {
                sum+=out_w[j][i+1]*out_error[j];
            }
            hide_error[i]=sum*out_x[i+1]*(1d-out_x[i+1]);
        }
    }

    //更新权重矩阵
    public void update_Weight(double [][]weight,double []err,double []x){
        double newWeight=0.0d;
        for (int i = 0; i <weight.length ; i++) {
            for (int j = 0; j <weight[i].length ; j++) {
                newWeight=rate*err[i]*x[j];
                //少了加原来weight
                weight[i][j]=weight[i][j]+newWeight;
            }
        }
    }

    //2.2 forward 输入-----隐含层输出-----输出层输出

    public void forward(double []x, double [] output){
        get_net_out(x,hide_w,out_x);
        get_net_out(out_x,out_w,output);
    }

    //单个节点的输出
    public double get_Node_put(double[] x,double[]w){
        double sum=0.0d;
        for (int i = 0; i <x.length ; i++) {
            sum+=x[i]*w[i];
        }
        return  1d/(1d+Math.exp(-sum));
    }

    public void get_net_out(double []x,double[][]w,double[] net_out){
        for (int i = 0; i <w.length ; i++) {
            net_out[i+1]=get_Node_put(x,w[i]);
        }
        net_out[0]=1d;
    }

    public void  predict(double[] data,double[] output){

        double []output1=new double[out_w.length+1];
        setHide_x(data);
        forward(hide_x,output1);
        System.arraycopy(output1,1,output,0,output.length);
    }




}
