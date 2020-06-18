package com.example.gan_mnist;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;


import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;


public class MainActivity extends AppCompatActivity {

    ImageView Gan_image;
    Button predict_btn, generate_btn;
    TextView Prediction_view;
    float[][] x_input;
    float[][] mnistOutput;
    Bitmap generated_image;
    String modelFile="converted_model.lite";
    String mnist_model = "mnist.lite";
    Interpreter tflite_gan, tflite_mnist;

    ByteBuffer input_data, output_img;


    private static final int NUMBER_LENGTH = 10;

    // Specify the input size
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 28;
    private static final int DIM_IMG_SIZE_Y = 28;
    private static final int DIM_PIXEL_SIZE = 1;

    // Number of bytes to hold a float (32 bits / float) / (8 bits / byte) = 4 bytes / float
    private static final int BYTE_SIZE_OF_FLOAT = 4;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        Gan_image=findViewById(R.id.GAN_image);
        predict_btn=findViewById(R.id.predict);
        generate_btn=findViewById(R.id.generate);
        Prediction_view=findViewById(R.id.result);

        generate_btn.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onClick(View v) {
                ByteBuffer output_data= Generate_Image();
                generated_image = getBitmap(output_data);
                Gan_image.setImageBitmap(Bitmap.createScaledBitmap(generated_image,300,300,false));
            }
        });


        predict_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                float result = Predict_digit(generated_image);
                Prediction_view.setText((int) result);

            }
        });
        try {
            tflite_gan = new Interpreter(loadModelFile(this, modelFile));
            tflite_mnist=new Interpreter(loadModelFile(this,mnist_model));

        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private ByteBuffer Generate_Image(){
        float[][] input = generate_latent_points(25,100);
        input_data = ByteBuffer.allocate( 100 * 4);
        input_data.order(ByteOrder.nativeOrder());
        output_img = ByteBuffer.allocate(25*28 * 28 * 4);
        output_img.order(ByteOrder.nativeOrder());

        input_data.clear();
        output_img.clear();
        input_data.rewind();

        for (int i=0;i<25;i++) {
            input_data.putFloat(input[0][i]);
        }

        tflite_gan.run(x_input, output_img);
        output_img.rewind();

        return output_img;

    }

    private Bitmap getBitmap(ByteBuffer output){

        Bitmap bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.RGB_565);
        int [] pixels = new int[28*28];
        for (int i = 0; i < 28*28; i++) {
            int a = 0xFF;
            float r = output.getFloat() * 255.0f;
            pixels[i] = a << 24 |  (((int) r)<<8);
        }
        bitmap.setPixels(pixels, 0, 28, 0, 0, 28, 28);
        return bitmap;
    }

    public MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[][] generate_latent_points(int n, int dim){
        Random rn = new Random();
        x_input=new float[1][100];
        for(int i =0; i<100; i ++){

            float num=rn.nextFloat();
            x_input[0][i]=num;
        }
        Log.d("INPUT ARRAY",Float.toString(x_input[0][99]));
        return x_input;
    }

    public float Predict_digit(Bitmap bitmap){
        float ans=0;
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                        BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        inputBuffer.order(ByteOrder.nativeOrder());
        mnistOutput = new float[DIM_BATCH_SIZE][NUMBER_LENGTH];
        int[] pixels = new int[28 * 28];

        // Load bitmap pixels into the temporary pixels variable
        bitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);

        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixels
            int pixel = pixels[i];
            int channel = pixel & 0xff;
            inputBuffer.putFloat(0xff - channel);
        Log.d("MNIST", "Created a Tensorflow Lite MNIST Classifier.");

        tflite_mnist.run(inputBuffer, mnistOutput);

        ans=postprocess(mnistOutput);

        }
        return ans;
    }

        float postprocess(float[][] mnistoutput){
            float answer=0;

            for (int m = 0; m < mnistoutput[0].length; m++) {
                float value = mnistoutput[0][m];
                Log.d("MNIST OUTPUT", "Output for " + Integer.toString(m) + ": " + Float.toString(value));
                // Check if this number is the one we care about. If yes, return the index
                if (value == 1f) {
                    answer=m;
                }
                else {
                    answer= -1;
                }
            }
            return answer;
        }

}