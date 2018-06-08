package org.deeplearning4j.solutions.earlystopping;

import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
 *
 * The first input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image.
 *
 * Look for LAB STEP below. Uncomment to proceed.
 * 1. valuate and save model if the accuracy is higher than the former
 *
 */

public class MnistMLP
{
    private static Logger log = LoggerFactory.getLogger(MnistMLP.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int numClasses = 10; // number of output classes
        int batchSize = 800; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 50; // number of epochs to perform
        double learningRate = 0.1; // learning rate
        String modelSavedPath = new ClassPathResource("earlystopping").getFile().getAbsolutePath() + "/mnistEpoch";

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        log.info("Build model....");

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nIn(100)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(150));

        log.info("Train model....");

        double accuracyMax = -1.0;
        double evalStep = 10;

        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);

            //Test and save model if accuracy lower
            Evaluation eval = new Evaluation(numClasses);

            while (mnistTest.hasNext())
            {
                DataSet next = mnistTest.next();
                INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
                eval.eval(next.getLabels(), output); //check the prediction against the true class
            }

            /*
		    #### LAB STEP 1 #####
            Evaluate and save model if the accuracy is higher than the former
            */

            if (i % evalStep == 0) {
                if (eval.accuracy() > accuracyMax) {
                    String modelSaveAs = modelSavedPath + Integer.toString(i) + ".zip";
                    accuracyMax = eval.accuracy();

                    log.info("Accuracy: " + accuracyMax);
                    log.info("Save model as " + modelSaveAs);
                    ModelSerializer.writeModel(model, modelSaveAs, false);
                } else {
                    log.info("Accuracy: " + eval.accuracy());
                }
            }

            mnistTrain.reset();
            mnistTest.reset();
        }


        //Test and save model if accuracy lower
        Evaluation eval = new Evaluation(numClasses);
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());

        /*
		    #### LAB STEP 1 #####
            Evaluate and save model if the accuracy is higher than the former
         */

        if (eval.accuracy() > accuracyMax) {
            String modelSaveAs = modelSavedPath + Integer.toString(numEpochs) + ".zip";
            log.info("Accuracy: " + eval.accuracy());
            log.info("Save model as " + modelSaveAs);
            ModelSerializer.writeModel(model, modelSaveAs, false);
        }




    }
}



