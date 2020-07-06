package com.example.gan_mnist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MnistClassifier {

    Interpreter tflite_mnist;
    String mnist_model="mnist.tflite";
    private Context context;

    private static final int NUMBER_LENGTH = 10;

    // Specify the input size
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 28;
    private static final int DIM_IMG_SIZE_Y = 28;
    private static final int DIM_PIXEL_SIZE = 1;

    // Number of bytes to hold a float (32 bits / float) / (8 bits / byte) = 4 bytes / float
    private static final int BYTE_SIZE_OF_FLOAT = 4;


    public MnistClassifier(Context context){
        this.context=context;
    }

    public void loadMnistModel(){

        try {
            tflite_mnist = new Interpreter(loadModelFile(context, mnist_model));
        }
        catch (IOException e) {
            e.printStackTrace();
        }

    }

    private MappedByteBuffer loadModelFile(Context c, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = c.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Bitmap ImageProcessing(Bitmap colored_bitmap){

        //Create a Black & white image by taking HSV values from RGB Image
        Bitmap bwBitmap = Bitmap.createBitmap( colored_bitmap.getWidth(),colored_bitmap.getHeight(), Bitmap.Config.RGB_565 );
        float[] hsv = new float[ 3 ];
        for( int col = 0; col < colored_bitmap.getWidth(); col++ ) {
            for( int row = 0; row < colored_bitmap.getHeight(); row++ ) {
                Color.colorToHSV( colored_bitmap.getPixel( col, row ), hsv );
                if( hsv[ 2 ] > 0.5f ) {
                    bwBitmap.setPixel( col, row, 0xff000000 );
                } else {
                    bwBitmap.setPixel( col, row, 0xffffffff );

                }
            }
        }
        return bwBitmap;
    }

    public ByteBuffer GetInput(Bitmap bitmap){

        //Get the black and white image from the colored bitmap
        Bitmap black_and_white=ImageProcessing(bitmap);

        //Allocate the buffers for input into the classifier
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        inputBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[28 * 28];
        // Load bitmap pixels into the temporary pixels variable
        black_and_white.getPixels(pixels, 0, 28, 0, 0, 28, 28);
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixels
            int pixel = pixels[i];
            int channel = pixel & 0xff;
            inputBuffer.putFloat(0xff - channel);
        }
        return inputBuffer;

    }


    public int Classify(ByteBuffer input){

        //Output is stored in a float array
        float[][] mnistOutput = new float[DIM_BATCH_SIZE][NUMBER_LENGTH];
        tflite_mnist.run(input, mnistOutput);
        float min_probability = Float.MIN_VALUE;
        int digit = 0;
        for (int i = 0; i < mnistOutput[0].length; i++) {
            float value = mnistOutput[0][i];
            if(value > min_probability){
                digit=i;
            }

    }
        return digit;

    }
}
