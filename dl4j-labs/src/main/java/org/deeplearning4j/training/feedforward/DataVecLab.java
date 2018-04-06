package org.deeplearning4j.training.feedforward;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * DataVecLab - Do multi-class classification for iris flowers with 4 features
 * There are 3 types of classes as output
 * Example:
 * Feature 1, Feature 2, Feature 3, Feature 4, Label
 * 5.1, 3.5, 1.4, 0.2, 0
 *
 * Look for LAB STEP below. Uncomment to proceed.
 * 1. Evaluate the network
 * 2. Tune the network
 *
 * @author Adam Gibson
 */

public class DataVecLab
{

    private static Logger log = LoggerFactory.getLogger(DataVecLab.class);

    public static void main(String[] args) throws  Exception
    {
        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        File filePath = new ClassPathResource("text/iris.txt").getFile();
        int numLinesToSkip = 0;

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip);
        recordReader.initialize(new FileSplit(filePath));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        DataSet allData = iterator.next(); //only one dataset in this case
        allData.shuffle();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();


        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        //DataNormalization normalizer = new NormalizerStandardize();
        //normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        //normalizer.transform(trainingData);     //Apply normalization to the training data
        //normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        int numInputs = 4;
        int numOutputs = 3;
        int seed = 123;
        int epoch = 20;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.01, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(10)
                        .nOut(numOutputs)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        /*
        Create a web based UI server to show progress as the network trains
        The Listeners for the model are set here as well
        One listener to pass stats to the UI
        and a Listener to pass progress info to the console
        */
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1), new StatsListener(storage, 1));

        for(int i = 0 ; i < epoch; ++i)
        {
            model.fit(trainingData);

            Thread.sleep(100);
        }

         /*
		#### LAB STEP 1 #####
		Evaluate the model on the test set
        */
        /*
        Evaluation eval = new Evaluation(numOutputs);
        INDArray predicted = model.output(testData.getFeatureMatrix());
        INDArray labels = testData.getLabels();

        for(int i = 0; i < labels.rows(); ++i)
        {
            System.out.println(labels.getRow(i) + " " +  predicted.getRow(i));
        }


        eval.eval(labels, predicted);
        log.info(eval.stats());
        */

        /*
		#### LAB STEP 2 #####
		Tune the network
        */

    }

}
