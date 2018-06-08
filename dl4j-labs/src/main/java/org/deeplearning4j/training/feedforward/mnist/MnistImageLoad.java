package org.deeplearning4j.training.feedforward.mnist;

import org.nd4j.linalg.io.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**

 *  This examples builds on the MnistImagePipelineExample
 *  by loading the trained network
 *
 * To run this sample, you must have
 * (1) save trained mnist model
 * (2) test image
 *
 *  Look for LAB STEP below. Uncomment to proceed.
 *  1. Load the saved model
 *  2. Load an image for testing
 *  3. [Optional] Preprocessing to 0-1 or 0-255
 *  4. Pass through the neural net for prediction
 */

public class MnistImageLoad
{
    private static Logger log = LoggerFactory.getLogger(org.deeplearning4j.solutions.feedforward.mnist.MnistImageLoad.class);

    public static void main(String[] args) throws Exception
    {
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;


        File modelSave =  new ClassPathResource("mnist_png/trained_mnist_model.zip").getFile();
        File imageToTest = new ClassPathResource("mnist_png/test.png").getFile();

        /*
		#### LAB STEP 1 #####
		Load the saved model
        */
        MultiLayerNetwork model = null;
        /*MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);*/

        /*
		#### LAB STEP 2 #####
		Load an image for testing
        */
        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = null;
        /*NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        // Get the image into an INDarray
        INDArray image = loader.asMatrix(imageToTest);
        */

        /*
		#### LAB STEP 3 #####
		[Optional] Preprocessing to 0-1 or 0-255
        */
        /*
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        */


        /*
		#### LAB STEP 4 #####
		[Optional] Pass to the neural net for prediction
        */
        /*
        INDArray output = model.output(image);

        log.info("Label: " + output.toString());
        */
    }

}
