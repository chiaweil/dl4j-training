package org.deeplearning4j.solutions.convolution.objectdetection;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * TinyYoloPrediction trained from Pascal VOC
 * Labels at https://github.com/allanzelener/YAD2K/blob/master/model_data/pascal_classes.txt
 */
public class TinyYoloPrediction
{
    private ComputationGraph preTrained;
    private static final TinyYoloPrediction INSTANCE = new TinyYoloPrediction();
    private List<DetectedObject> predictedObjects;
    private HashMap<Integer, String> map;
    private static final int width = 416;
    private static final int height = 416;
    private static int gridWidth = 13;
    private static int gridHeight = 13;
    private static double detectionThreshold = 0.5;

    public static TinyYoloPrediction getINSTANCE()
    {
        return INSTANCE;
    }

    private TinyYoloPrediction() {
        try {
            preTrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
            prepareLabels();
            System.out.println(preTrained.summary());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private INDArray prepareImage(opencv_core.Mat file, int width, int height) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        INDArray indArray = loader.asMatrix(file);
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }

    private void prepareLabels() {
        if (map == null) {
            String s = "aeroplane\n" + "bicycle\n" + "bird\n" + "boat\n" + "bottle\n" + "bus\n" + "car\n" +
                    "cat\n" + "chair\n" + "cow\n" + "diningtable\n" + "dog\n" + "horse\n" + "motorbike\n" +
                    "person\n" + "pottedplant\n" + "sheep\n" + "sofa\n" + "train\n" + "tvmonitor";
            String[] split = s.split("\\n");
            int i = 0;
            map = new HashMap<>();
            for (String s1 : split) {
                map.put(i++, s1);
            }
        }
    }

    private static void removeObjectsIntersectingWithMax(ArrayList<DetectedObject> detectedObjects, DetectedObject maxObjectDetect) {
        double[] bottomRightXY1 = maxObjectDetect.getBottomRightXY();
        double[] topLeftXY1 = maxObjectDetect.getTopLeftXY();
        List<DetectedObject> removeIntersectingObjects = new ArrayList<>();
        for (DetectedObject detectedObject : detectedObjects) {
            double[] topLeftXY = detectedObject.getTopLeftXY();
            double[] bottomRightXY = detectedObject.getBottomRightXY();
            double iox1 = Math.max(topLeftXY[0], topLeftXY1[0]);
            double ioy1 = Math.max(topLeftXY[1], topLeftXY1[1]);

            double iox2 = Math.min(bottomRightXY[0], bottomRightXY1[0]);
            double ioy2 = Math.min(bottomRightXY[1], bottomRightXY1[1]);

            double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

            double box1_area = (bottomRightXY1[1] - topLeftXY1[1]) * (bottomRightXY1[0] - topLeftXY1[0]);
            double box2_area = (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

            double union_area = box1_area + box2_area - inter_area;
            double iou = inter_area / union_area;


            if (iou > 0.5) {
                removeIntersectingObjects.add(detectedObject);
            }

        }
        detectedObjects.removeAll(removeIntersectingObjects);
    }


    public static void setDetectionThreshold(double input)
    {
        detectionThreshold = input;
    }

    public void markWithBoundingBox(opencv_core.Mat file, int imageWidth, int imageHeight, boolean newBoundingBox) throws Exception {


        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) preTrained.getOutputLayer(0);

        if (newBoundingBox)
        {
            INDArray indArray = prepareImage(file, width, height);
            INDArray results = preTrained.outputSingle(indArray);
            predictedObjects = outputLayer.getPredictedObjects(results, detectionThreshold);

            //System.out.println("predicted objects: " + predictedObjects.size());
            markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);
        }
        else
        {
            markWithBoundingBox(file, gridWidth, gridHeight, imageWidth, imageHeight);


        }
    }

    private void markWithBoundingBox(opencv_core.Mat file, int gridWidth, int gridHeight, int w, int h)
    {
        if (predictedObjects == null)
        {
            return;
        }

        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);

        while (!detectedObjects.isEmpty())
        {
            Optional<DetectedObject> max = detectedObjects.stream().max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));

            if (max.isPresent())
            {
                DetectedObject maxObjectDetect = max.get();
                removeObjectsIntersectingWithMax(detectedObjects, maxObjectDetect);
                detectedObjects.remove(maxObjectDetect);
                markWithBoundingBox(file, gridWidth, gridHeight, w, h, maxObjectDetect);
            }
        }
    }

    private void markWithBoundingBox(opencv_core.Mat file, int gridWidth, int gridHeight, int w, int h, DetectedObject obj)
    {

        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();
        int predictedClass = obj.getPredictedClass();

        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / gridHeight);

        rectangle(file, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
        putText(file, map.get(predictedClass), new opencv_core.Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, opencv_core.Scalar.GREEN);
    }
}
