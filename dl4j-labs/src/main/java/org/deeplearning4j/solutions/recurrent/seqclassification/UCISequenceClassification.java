package org.deeplearning4j.solutions.recurrent.seqclassification;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * @author Alex Black

 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * Look for LAB STEP below. Uncomment to proceed.
 * 1. Download and prepare the data (in downloadUCIData() method)
 *    1.1 Split each sequential data into row data form, one time step per row. Each sequential data assigned with an label.
 *    1.2 Split the 600 sequences into train set of size 450, and test set of size 150
 *        Write the data into a format suitable for loading using the CSVSequenceRecordReader for sequence classification
 *        This format: one time series per file, and a separate file for the labels.
 *        For example, train/features/0.csv is the features using with the labels file train/labels/0.csv
 *        Because the data is a univariate time series, we only have one column in the CSV files. Normally, each column
 *        would contain multiple values - one time step per row.
 *        Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value
 *
 * 2. Load the testing data using CSVSequenceRecordReader (to load/parse the CSV files) and SequenceRecordReaderDataSetIterator
 *    (to convert it to DataSet objects, ready to train)
 *    For more details on this step, see: http://deeplearning4j.org/usingrnns#data
 *
 * 3. Normalize the data. The raw data contain values that are too large for effective training, and need to be normalized.
 *    Normalization is conducted using NormalizerStandardize, based on statistics (mean, st.dev) collected on the training
 *    data only. Note that both the training data and test data are normalized in the same way.
 *
 * 4. Configure the network
 *    The data set here is very small, so we can't afford to use a large network with many parameters.
 *    We are using one small LSTM layer and one RNN output layer
 *
 * 5. Train the network for 40 epochs
 *    At a numbered steps of epochs, evaluate and print the accuracy and f1 on the test set
 *
 * 6. [Extra step] Tune the network
 */

public class UCISequenceClassification
{
    private static File baseDir = new File("dl4j-labs/src/main/resources/UCI/");

    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");

    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");


    public static void main(String[] args) throws Exception
    {

        /*
		#### LAB STEP 1 #####
		Download and prepare the data
        */
        final int miniBatchSize = 10;
        final int numLabelClasses = 6;

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
		#### LAB STEP 2 #####
		Load the testing data
		Note that we have 150 training files for features: test/features/0.csv through test/features/149.csv
        */
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
		#### LAB STEP 3 #####
		Normalize the data
        */
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData); //Collect training data statistics

        //Use previously collected statistics to normalize on-the fly.EachDataSet returned by 'trainData' iterator will be normalized.
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        trainData.reset();
        /*
		#### LAB STEP 4 #####
		Configure the network
        */
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)          //Random number generator seed for improved repeatability.
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.005, 0.9))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder()
                        .activation(Activation.TANH)
                        .nIn(1)
                        .nOut(10)
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(10)
                        .nOut(numLabelClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);

        //set server listeners
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();

        server.attach(storage);

        network.setListeners(new StatsListener(storage, 10));


        // ----- Train the network, evaluating the test set performance at each epoch -----
        /*
		#### LAB STEP 5 #####
		Train network
        */
        int nEpochs = 40;
        int evaluateStep = 5;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";

        for (int i = 0; i < nEpochs; ++i)
        {
            network.fit(trainData);

            //Evaluate on the test set:

            if(i % evaluateStep == 0)
            {
                Evaluation evaluation = network.evaluate(testData);
                System.out.println(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
                testData.reset();
            }

            trainData.reset();
        }

        Evaluation evaluation = network.evaluate(testData);
        System.out.println(evaluation.stats());

        System.out.println("----- Example Complete -----");

        /*
		#### LAB STEP 6 #####
		Tune the network
        */

    }

    //This method downloads the data and converts the "one time series per line" format into
    //CSV sequence format that DataVec (CSVSequenceRecordRead) and DL4J can read
    private static void downloadUCIData() throws Exception
    {
        if (baseDir.exists()) {
            System.out.println("Data Directory exist. Not downloading");
            return;
        }

        System.out.println("Downloading data");
        System.out.println("Path: " + baseDir.getAbsoluteFile());


        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        /*
		#### LAB STEP 1.1 #####
		Split each sequential data into row data form, one time step per row. Each sequential data assigned with an label.
        */

        String[] lines = data.split("\n");

        int lineCount = 0;

        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines)
        {
            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            /*
            #### LAB STEP 1.1 #####
            Split each sequential data into row data form, one time step per row. Each sequential data assigned with an label.
            */
            String transposed = line.replaceAll(" +", "\n");
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));



        //Create directories
        File[] directoryArray = {baseDir, baseTrainDir, featuresDirTrain, labelsDirTrain, baseTestDir, featuresDirTest, labelsDirTest};
        for (File direc : directoryArray) {
            direc.mkdir();
        }

        /*
		#### LAB STEP 1.2 #####
		Split the 600 sequences into train set of size 450, and test set of size 150
        */
        int nTrain = 450;  //Enter number of training samples  //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels)
        {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;

            if (trainCount < nTrain)
            {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");

                trainCount++;
            }
            else
            {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");

                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }

    }

}
