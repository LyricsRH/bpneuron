package edu.packt.neuralnet.learn;

import java.util.ArrayList;

import edu.packt.neuralnet.InputLayer;
import edu.packt.neuralnet.NeuralNet;
import edu.packt.neuralnet.Neuron;

public abstract class Training {

	private int epochs;
	private double error;
	private double mse;

	public enum TrainingTypesENUM {
		PERCEPTRON, ADALINE;
	}

	public NeuralNet train(NeuralNet n) {
		
		ArrayList<Double> inputWeightIn = new ArrayList<Double>();


		//一行是一组训练集，，共有rows组，每组由cols个输入
		int rows = n.getTrainSet().length;
		int cols = n.getTrainSet()[0].length;

		//迭代次数
		while (this.getEpochs() < n.getMaxEpochs()) {

			double estimatedOutput = 0.0;
			double realOutput = 0.0;

			//遍历每组训练集
			for (int i = 0; i < rows; i++) {

				double netValue = 0.0;
				//遍历每个输入
				for (int j = 0; j < cols; j++) {
					//每个输入对应一个输入层神经元
					inputWeightIn = n.getInputLayer().getListOfNeurons().get(j)
							.getListOfWeightIn();
					double inputWeight = inputWeightIn.get(0);//只有一个输出的目的神经元，，，有多个要遍历get(i)
					netValue = netValue + inputWeight * n.getTrainSet()[i][j];
				}
				//这是简单只有3个输入层神经元，没有隐藏层，直接到输出层一个神经元

				//选择激活函数，并计算权值和的处理结果
				estimatedOutput = this.activationFnc(n.getActivationFnc(),
						netValue);

				//获得真实值
				realOutput = n.getRealOutputSet()[i];

				//计算实际值与训练值的误差
				this.setError(realOutput - estimatedOutput);

				// System.out.println("Epoch: "+this.getEpochs()+" / Error: " + this.getError());

				//有个阙值，误差大于阙值才进行权值更新
				if (Math.abs(this.getError()) > n.getTargetError()) {
					// fix weights
					InputLayer inputLayer = new InputLayer();
					inputLayer.setListOfNeurons(this.teachNeuronsOfLayer(cols,
							i, n, netValue));
					n.setInputLayer(inputLayer);
				}

			}

			this.setMse(Math.pow(realOutput - estimatedOutput, 2.0));
			n.getListOfMSE().add(this.getMse());

			this.setEpochs(this.getEpochs() + 1);

		}

		n.setTrainingError(this.getError());

		return n;
	}


	//返回输入层更新过权重的神经元列表  （输入层神经元个数，第几个样本为了得到输入值，神经网络，该神经元的输出值）
	private ArrayList<Neuron> teachNeuronsOfLayer(int numberOfInputNeurons,
			int line, NeuralNet n, double netValue) {
		ArrayList<Neuron> listOfNeurons = new ArrayList<Neuron>();
		ArrayList<Double> inputWeightsInNew = new ArrayList<Double>();
		ArrayList<Double> inputWeightsInOld = new ArrayList<Double>();

		for (int j = 0; j < numberOfInputNeurons; j++) {
			inputWeightsInOld = n.getInputLayer().getListOfNeurons().get(j)
					.getListOfWeightIn();
			double inputWeightOld = inputWeightsInOld.get(0);

			//只有一个输出神经元，
			inputWeightsInNew.add( this.calcNewWeight(n.getTrainType(),
					inputWeightOld, n, this.getError(),
					n.getTrainSet()[line][j], netValue) );

			Neuron neuron = new Neuron();
			//更新过权重的神经元
			neuron.setListOfWeightIn(inputWeightsInNew);
			listOfNeurons.add(neuron);
			//变成空的
			inputWeightsInNew = new ArrayList<Double>();
		}

		return listOfNeurons;

	}
	//计算单个权值
	private double calcNewWeight(TrainingTypesENUM trainType,
			double inputWeightOld, NeuralNet n, double error,
			double trainSample, double netValue) {
		switch (trainType) {
		case PERCEPTRON:
			return inputWeightOld + n.getLearningRate() * error * trainSample;
		case ADALINE:
			return inputWeightOld + n.getLearningRate() * error * trainSample
					* derivativeActivationFnc(n.getActivationFnc(), netValue);
		default:
			throw new IllegalArgumentException(trainType
					+ " does not exist in TrainingTypesENUM");
		}
	}

	public enum ActivationFncENUM {
		STEP, LINEAR, SIGLOG, HYPERTAN;
	}

	private double activationFnc(ActivationFncENUM fnc, double value) {
		switch (fnc) {
		case STEP:
			return fncStep(value);
		case LINEAR:
			return fncLinear(value);
		case SIGLOG:
			return fncSigLog(value);
		case HYPERTAN:
			return fncHyperTan(value);
		default:
			throw new IllegalArgumentException(fnc
					+ " does not exist in ActivationFncENUM");
		}
	}

	public double derivativeActivationFnc(ActivationFncENUM fnc, double value) {
		switch (fnc) {
		case LINEAR:
			return derivativeFncLinear(value);
		case SIGLOG:
			return derivativeFncSigLog(value);
		case HYPERTAN:
			return derivativeFncHyperTan(value);
		default:
			throw new IllegalArgumentException(fnc
					+ " does not exist in ActivationFncENUM");
		}
	}

	private double fncStep(double v) {
		if (v >= 0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
	private double fncLinear(double v) {
		return v;
	}
	private double fncSigLog(double v) {
		return 1.0 / (1.0 + Math.exp(-v));
	}
	private double fncHyperTan(double v) {
		return Math.tanh(v);
	}

	private double derivativeFncLinear(double v) {
		return 1.0;
	}
	private double derivativeFncSigLog(double v) {
		return v * (1.0 - v);
	}
	private double derivativeFncHyperTan(double v) {
		return (1.0 / Math.pow(Math.cosh(v), 2.0));
	}

	public void printTrainedNetResult(NeuralNet trainedNet) {

		int rows = trainedNet.getTrainSet().length;
		int cols = trainedNet.getTrainSet()[0].length;

		ArrayList<Double> inputWeightIn = new ArrayList<Double>();

		for (int i = 0; i < rows; i++) {
			double netValue = 0.0;
			for (int j = 0; j < cols; j++) {
				inputWeightIn = trainedNet.getInputLayer().getListOfNeurons()
						.get(j).getListOfWeightIn();
				double inputWeight = inputWeightIn.get(0);
				netValue = netValue + inputWeight
						* trainedNet.getTrainSet()[i][j];

				System.out.print(trainedNet.getTrainSet()[i][j] + "\t");
			}

			double estimatedOutput = this.activationFnc(
					trainedNet.getActivationFnc(), netValue);

			System.out.print(" NET OUTPUT: " + estimatedOutput + "\t");
			System.out.print(" REAL OUTPUT: "
					+ trainedNet.getRealOutputSet()[i] + "\t");
			double error = estimatedOutput - trainedNet.getRealOutputSet()[i];
			System.out.print(" ERROR: " + error + "\n");

		}

	}

	public int getEpochs() {
		return epochs;
	}

	public void setEpochs(int epochs) {
		this.epochs = epochs;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	public double getMse() {
		return mse;
	}

	public void setMse(double mse) {
		this.mse = mse;
	}

}
