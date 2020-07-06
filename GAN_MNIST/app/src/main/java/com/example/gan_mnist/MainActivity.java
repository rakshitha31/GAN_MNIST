package com.example.gan_mnist;


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

import java.nio.ByteBuffer;


public class MainActivity extends AppCompatActivity {

    ImageView Gan_image;
    Button predict_btn, generate_btn;
    TextView Prediction_view;
    Bitmap image;


    @Override
    protected  void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        Gan_image=findViewById(R.id.GAN_image);
        predict_btn=findViewById(R.id.predict);
        generate_btn=findViewById(R.id.generate);
        Prediction_view=findViewById(R.id.result);

        //Create objects of classes
        final GAN gan=new GAN(this);
        final MnistClassifier classify=new MnistClassifier(this);

        generate_btn.setOnClickListener(new View.OnClickListener() {

            @RequiresApi(api = Build.VERSION_CODES.O)
            @Override
            public void onClick(View v) {
                gan.loadGanModel();
                image=gan.generateImage();
                Gan_image.setImageBitmap(Bitmap.createScaledBitmap(image,300,300,false));
            }
        });

        predict_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                classify.loadMnistModel();
                ByteBuffer mnist_input=classify.GetInput(image);
                float result=classify.Classify(mnist_input);
                Prediction_view.setText(String.valueOf(( result)));
            }
        });
    }




}