package com.example.gan_mnist;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ColorSpace;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.util.Log;
import android.view.View;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;


public class MainActivity extends AppCompatActivity {

    public static final String MODEL_NAME="converted_model.lite";
    ImageView Gan_image;
    Button predict_btn, generate_btn;
    TextView Prediction_view;
    float[] x_input;
    String modelFile="converted_model.lite";
    Interpreter tflite;
    ByteBuffer input_data, output_img;



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
            @Override
            public void onClick(View v) {
                float[] input = generate_latent_points(25,100);
                input_data = ByteBuffer.allocate( 100 * Float.BYTES);
                input_data.order(ByteOrder.nativeOrder());

                output_img = ByteBuffer.allocate(1* 28 * 28 * Float.BYTES);
                output_img.order(ByteOrder.nativeOrder());

                input_data.clear();
                output_img.clear();
                input_data.rewind();

                for (int i=0;i<25;i++) {
                        input_data.putFloat(input[i]);
                }

                Log.d("INPUT", String.valueOf(input_data));
                tflite.run(input_data, output_img);
                output_img.rewind();
                Bitmap bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
                bitmap.copyPixelsFromBuffer(output_img);



                Log.d("WHOAAA", String.valueOf(output_img.asFloatBuffer()));
                Gan_image.setImageBitmap(Bitmap.createScaledBitmap(bitmap,250,250,false));
            }
        });


        try {
            tflite = new Interpreter(loadModelFile(this, modelFile));
        }
        catch (IOException e) {
            e.printStackTrace();
        }

    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[] generate_latent_points(int n, int dim){
        Random rn = new Random();
        x_input=new float[100];
        for(int i =0; i<100; i ++){


                float num=rn.nextFloat();
                x_input[i]=num;


        }
        Log.d("Holla Holla",Float.toString(x_input[0]));
    return x_input;
    }


}