package org.deeplearning4j.solutions.feedforward;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * "Linear" Data Classification Example
 *
 */

public class MLPClassifierLinear
{

    static final Logger log = LoggerFactory.getLogger(MLPClassifierLinear.class);

    public static void main(String[] args) throws Exception
    {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        int batchSize = 50;
        int seed = 123;
        double learningRate = 0.005;

        //Number of epochs(full passes of the data)
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 5;

        final String fileNameTrain = new ClassPathResource("/classification/linear_data_train.csv").getFile().getPath();
        final String fileNameTest = new ClassPathResource("/classification/linear_data_eval.csv").getFile().getPath();

        //load the training data
        RecordReader rrTrain = new CSVRecordReader();
        rrTrain.initialize(new FileSplit(new File(fileNameTrain)));

        DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(rrTrain, batchSize, 0, 2);

        //load the test/evaluation data
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(fileNameTest)));

        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        //Build Model Configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        //Server
        StatsStorage statsStorage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(statsStorage);

        model.setListeners(new StatsListener(statsStorage, 10));

        //Train Model
        log.info("Train Model");
        for(int i = 0; i < nEpochs; ++i)
        {
            model.fit(dataIterTrain);
        }

        //Evaluate Model
        log.info("Evaluate Model");
        Evaluation eval = new Evaluation(numOutputs);

        while(dataIterTest.hasNext())
        {
            DataSet dataTest = dataIterTest.next();
            INDArray featureTest = dataTest.getFeatureMatrix();
            INDArray predicted = model.output(featureTest, false);

            eval.eval(dataTest.getLabels(), predicted);
        }

        log.info(eval.stats());

    }

}

