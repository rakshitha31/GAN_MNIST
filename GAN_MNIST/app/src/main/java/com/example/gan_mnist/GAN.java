package com.example.gan_mnist;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
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
import java.util.Random;

public class GAN {

    Interpreter tflite_gan;
    String gan_model="gan_model.tflite";
    private Context context;

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    // Number of bytes to hold a float (32 bits / float) / (8 bits / byte) = 4 bytes / float
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    private static final int N= 1;
    private static final int DIM = 100;

    public GAN(Context context){
        this.context=context;
    }

    public void loadGanModel(){

        try {
            tflite_gan = new Interpreter(loadModelFile(context, gan_model));
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

    public Bitmap generateImage(){

        float[][] input = generateLatentPoints(N,DIM);
        ByteBuffer input_data = ByteBuffer.allocate( 100 * BYTE_SIZE_OF_FLOAT);
        input_data.order(ByteOrder.nativeOrder());

        ByteBuffer gan_output = ByteBuffer.allocate(1 *28 * 28 * 1 * BYTE_SIZE_OF_FLOAT);
        gan_output.order(ByteOrder.nativeOrder());

        input_data.clear();
        gan_output.clear();
        input_data.rewind();

        for (int i=0;i<100;i++) {
            input_data.putFloat(input[0][i]);
        }

        tflite_gan.run(input, gan_output);
        gan_output.rewind();
        Bitmap bitmap = Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT, Bitmap.Config.RGB_565);
        int [] pixels = new int[IMAGE_WIDTH * IMAGE_HEIGHT];
        for (int i = 0; i < 28*28; i++) {

            int a = 0xFF;
            float r = gan_output.getFloat() * 255.0f;
            pixels[i] = a << 24 |  (((int) r)<<8);
        }
        //This is a colored image which you can display
        bitmap.setPixels(pixels, 0, 28, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);



        return bitmap;

   //     return bwBitmap;
    }

    private float[][] generateLatentPoints(int n, int dim){
        Random rn = new Random();
        float[][] gan_input=new float[1][100];
        for(int i =0; i<100; i ++){

            float random_number=rn.nextFloat();
            gan_input[0][i]=random_number;
        }
        return gan_input;
    }



}
