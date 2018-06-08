package org.deeplearning4j.solutions.convolution.objectdetection;



import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.*;

import java.awt.event.KeyEvent;

/**
 * Yolo Detection with video
 *
 * Get Sample video from below
 * https://drive.google.com/open?id=1Xx52YgF4R6BUBjuVo0vWRfTTG9oOoZCs
 * https://drive.google.com/open?id=1qjEBc37E0OBLUdNOGYeHxHbRpPVoU2G0
 *
 */

public class VideoObjectDetection
{
    private static Thread thread;

    public static void main(String[] args) throws Exception {

        String videoPath = "test.mp4";
        FrameGrabber grabber = FrameGrabber.createDefault(videoPath);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());

        while (true)
        {
            try {
                Frame frame = grabber.grab();

                opencv_core.Mat matFrame = converter.convert(frame);

                //if a thread is null, create new thread
                if (thread == null)
                {
                    thread = new Thread(() ->
                    {
                        while (frame != null)
                        {
                            try
                            {
                                TinyYoloPrediction.getINSTANCE().markWithBoundingBox(matFrame, frame.imageWidth, frame.imageHeight, true);
                            }
                            catch (java.lang.Exception e)
                            {
                                throw new RuntimeException(e);
                            }
                        }
                    });
                    thread.start();
                }

                //continous get the max suppression of all the threads and display it.
                TinyYoloPrediction.getINSTANCE().markWithBoundingBox(matFrame, frame.imageWidth, frame.imageHeight, false);


                canvas.showImage(converter.convert(matFrame));

                KeyEvent t = canvas.waitKey(25);

                if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                    break;
                }
            }
            catch(FrameGrabber.Exception e)
            {
                break;
            }
        }

        canvas.dispose();

    }

}

