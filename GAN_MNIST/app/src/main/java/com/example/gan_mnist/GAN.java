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
    String gan_model="converted_model.lite";
    private Context context;

//save the context recievied via constructor in a local variable

    public GAN(Context context){
        this.context=context;
    }


    public void Load_Gan_model(){

        try {
            tflite_gan = new Interpreter(loadModelFile(context, gan_model));
        }
        catch (IOException e) {
            e.printStackTrace();
        }

    }

    public Bitmap Generate_Image(){

        float[][] input = Generate_latent_points(25,100);
        ByteBuffer input_data = ByteBuffer.allocate( 100 * 4);
        input_data.order(ByteOrder.nativeOrder());

        ByteBuffer output_img = ByteBuffer.allocate(25*28 * 28 * 4);
        output_img.order(ByteOrder.nativeOrder());

        input_data.clear();
        output_img.clear();
        input_data.rewind();

        for (int i=0;i<25;i++) {
            input_data.putFloat(input[0][i]);
        }

        Log.d("INPUT", String.valueOf(input_data));
        tflite_gan.run(input, output_img);
        output_img.rewind();
        Log.d("OUTPUT BUFFER", String.valueOf(output_img.get(60)));
        Bitmap bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.RGB_565);
        int [] pixels = new int[28*28];
        for (int i = 0; i < 28*28; i++) {

            int a = 0xFF;
            float r = output_img.getFloat() * 255.0f;
            pixels[i] = a << 24 |  (((int) r)<<8);
        }
        bitmap.setPixels(pixels, 0, 28, 0, 0, 28, 28);


        Bitmap bwBitmap = Bitmap.createBitmap( bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.RGB_565 );
        float[] hsv = new float[ 3 ];
        for( int col = 0; col < bitmap.getWidth(); col++ ) {
            for( int row = 0; row < bitmap.getHeight(); row++ ) {
                Color.colorToHSV( bitmap.getPixel( col, row ), hsv );
                if( hsv[ 2 ] > 0.5f ) {
                    bwBitmap.setPixel( col, row, 0xff000000 );
                } else {
                    bwBitmap.setPixel( col, row, 0xffffffff );

                }
            }
        }

        return bwBitmap;
    }

    private float[][] Generate_latent_points(int n, int dim){
        Random rn = new Random();
        float[][] gan_input=new float[1][100];
        for(int i =0; i<100; i ++){

            float num=rn.nextFloat();
            gan_input[0][i]=num;
        }
        Log.d("INPUT ARRAY",Float.toString(gan_input[0][99]));
        return gan_input;
    }


    private MappedByteBuffer loadModelFile(Context c, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = c.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }




}
