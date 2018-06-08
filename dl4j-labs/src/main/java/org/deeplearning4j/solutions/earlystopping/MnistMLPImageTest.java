package org.deeplearning4j.solutions.earlystopping;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;


/**
 * Testing image classification
 * image --> INDArray
 *
 * Look for LAB STEP below. Uncomment to proceed.
 * 1. Point to the model saved path and image to test
 * 2. Load the saved model
 * 3. Predict the label data
 */
public class MnistMLPImageTest
{
    private static Logger log = LoggerFactory.getLogger(MnistMLPImageTest.class);

    public static void main(String[] args) throws Exception
    {
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;

        /*
		#### LAB STEP 1 #####
		Download and prepare the data
		Point to the model saved path and image to test
        */
        File imageToTest = new ClassPathResource("earlystopping/3.pngearlystopping/3.png").getFile();
        File modelSave =  new ClassPathResource("earlystopping/mnistEpoch40.zip").getFile();

        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray
        INDArray image = loader.asMatrix(imageToTest);

        //preprocessing to 0 - 1. The same preprocessing during training
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        //reduce the rank
        INDArray arr = image.getRow(0).getRow(0);

        //reshape to suit MultiLayerNetwork data type
        INDArray featureMatrix = arr.reshape(1, height * width);

        /*
		#### LAB STEP 2 #####
		Load the saved model
        */
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);


        /*
		#### LAB STEP 3 #####
		Predict the label of the data
        */
        //predict the data
        INDArray output  = model.output(featureMatrix);

        //get the arg with maximum probability
        INDArray argMax = Nd4j.getExecutioner().exec(new IMax(output), 1);

        log.info("Label: " + argMax.getInt(0));

    }
}
