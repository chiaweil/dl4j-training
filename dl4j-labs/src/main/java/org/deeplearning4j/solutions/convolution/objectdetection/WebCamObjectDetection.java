package org.deeplearning4j.solutions.convolution.objectdetection;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.*;

import java.awt.event.KeyEvent;

/**
 * YOLO detection with camera
 * Note 1: Swap between camera by using createDefault parameter
 * Note 2: flip the camera if opening front camera
 */
public class WebCamObjectDetection
{
    private static Thread thread;

    public static void main(String[] args) throws Exception
    {

        //swap between camera with 0 -? on the parameter
        FrameGrabber grabber = FrameGrabber.createDefault(0);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();

        String winName = "Webcam Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());

        //Set Detection Threshold
        TinyYoloPrediction.getINSTANCE().setDetectionThreshold(0.4);

        while (true)
        {
            try {
                Frame frame = grabber.grab();

                opencv_core.Mat mt = converter.convert(frame);

                //Flip the camera if opening front camera
                //opencv_core.flip(mt, mt, 1);

                //if a thread is null, create new thread
                if (thread == null)
                {
                    thread = new Thread(() ->
                    {
                        while (frame != null)
                        {
                            try
                            {
                                TinyYoloPrediction.getINSTANCE().markWithBoundingBox(mt, frame.imageWidth, frame.imageHeight, true);
                            }
                            catch (java.lang.Exception e)
                            {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                    thread.start();
                }

                KeyEvent t = canvas.waitKey(25);

                if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                    break;
                }

                //continous get the max suppression of all the threads and display it.
                TinyYoloPrediction.getINSTANCE().markWithBoundingBox(mt, frame.imageWidth, frame.imageHeight, false);

                canvas.showImage(converter.convert(mt));


            }
            catch(FrameGrabber.Exception e)
            {
                break;
            }
        }

        canvas.dispose();
        grabber.close();

    }
}

