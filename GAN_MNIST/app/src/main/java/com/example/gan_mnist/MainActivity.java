package com.example.gan_mnist;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;


import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

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


public class MainActivity extends AppCompatActivity {

    ImageView Gan_image;
    Button predict_btn, generate_btn;
    TextView Prediction_view;

    Bitmap generated_image;
    ByteBuffer output_data, input_data;



    String mnist_model = "mnist.lite";
    Interpreter tflite_mnist;
    float[][] mnistOutput, latent_points;
    ByteBuffer output_img, mnist_input;



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

        final GAN_Digit Gan = new GAN_Digit();
        final DigitClassifier classify=new DigitClassifier();


        try {

            tflite_mnist=new Interpreter(loadModelFile(this,mnist_model));
            tflite_mnist=new Interpreter(loadModelFile(this,gan_model));

        }
        catch (
                IOException e) {
            e.printStackTrace();
        }

        generate_btn.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onClick(View v) {
                latent_points= Gan.generate_latent_points(25,100);
                output_img = ByteBuffer.allocate(1 * 28 * 28 * 4);
                output_img.order(ByteOrder.nativeOrder());
                tflite_gan.run(latent_points, output_img);
                output_data= Gan.Generate_Image();
                generated_image = Gan.getBitmap(output_data);
                Gan_image.setImageBitmap(Bitmap.createScaledBitmap(generated_image,300,300,false));
            }
        });


        predict_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mnist_input=classify.Classifier_input(generated_image);
                tflite_mnist.run(mnist_input, mnistOutput);
                float result= classify.Predict_digit(mnistOutput);

                Prediction_view.setText(String.valueOf(result));

            }
        });

    }









}