package com.example.dcrocco.camerarectangledetection;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static String TAG = "MainActivity";

    private JavaCameraView javaCameraView;

    private Mat imgSource;

    // Configuracion de Frames
    private int frame;
    private int maxFrames = 1;

    private int channelToExtract = 1;

    private int areaSize = 1000;

    private int thresholdMin = 0;
    private int thresholdMax = 255;
    private int thresholdLvl = 11;

    private List<MatOfPoint> squares = new ArrayList<>();

    private int maxSquares = 3;

    // Vector de colores
    private Scalar colorRed = new Scalar(255,0,0);
    private Scalar colorBlue = new Scalar(0,255,0);
    private Scalar colorGreen = new Scalar(0,0,255);
    private Scalar[] colorsVector = {colorRed, colorBlue, colorGreen};

    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status){
                case BaseLoaderCallback.SUCCESS:{
                    javaCameraView.enableView();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        javaCameraView = (JavaCameraView)findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        frame = 12;
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }
    }

    @Override
    protected void onPause(){
        super.onPause();
        if (javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if (javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume(){
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "  OpenCVLoader.initDebug(), not working.");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "  OpenCVLoader.initDebug(), working.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {
        imgSource.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Trabajamos cada 12 frames para optimizar la aplicacion y que a la vez se vea fluida
        if (frame == maxFrames){
            imgSource = inputFrame.rgba();

            findSquares(imgSource.clone(), squares);

            // Descartamos todos los otros rectangulos encontrados.
            // Solo nos quedamos con los primeros 3.
            squares = squares.subList(0, Math.min(squares.size(), maxSquares));

            int colorIndex = 0;

            // Iteramos por todos los rectangulos encontrados
            // como maximo seran 3 dado que descartamos los demas
            for (MatOfPoint mop: squares){

                // Tomamos el color del vectos de colores
                // tener en cuenta que el vector de colores se inicializo
                // solo con tres colores. !!
                Scalar color = colorsVector[colorIndex++];

                // Dibujamos el contorno para el rectangulo actual
                List<MatOfPoint> auxSquareList = new ArrayList<>();
                auxSquareList.add(mop);
                Imgproc.drawContours(imgSource, auxSquareList , -1, color);

                // Armo listado de MatOfPoint auxiliar para dibujar el triangulo
                // Vease que se toma luego de pointsMop los indices 0, 1 , 2 , 3
                // Siempre van a existir hasta 4 elementos dado que son los puntos
                // que componen el rectangulo
                List<MatOfPoint> auxTriangleList = new ArrayList<>();
                MatOfPoint auxMatOfPoint = new MatOfPoint();
                List<Point> auxPointList = new ArrayList<>();
                List<Point> pointsMop = mop.toList();
                auxPointList.add(pointsMop.get(0));
                auxPointList.add(new Point((pointsMop.get(1).x + pointsMop.get(2).x) / 2, (pointsMop.get(1).y + pointsMop.get(2).y) / 2));
                auxPointList.add(pointsMop.get(3));
                auxMatOfPoint.fromList(auxPointList);
                auxTriangleList.add(auxMatOfPoint);

                // Se dibuja el triangulo sobre el rectangulo
                Imgproc.fillPoly(imgSource, auxTriangleList , color);
            }
        }

        frame --;
        if (frame == 0){
            frame = maxFrames;
        }

        return imgSource;
    }

    private double angle( Point pt1, Point pt2, Point pt0 ) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }

    // Devuelve la secuencia de rectangulos detectados en la imagen
    private void findSquares( Mat image, List<MatOfPoint> fsquares ) {

        squares.clear();

        // Achicamos la imagen para optimizar procesamiento
        Mat smallerImg=new Mat(new Size(image.width()/2, image.height()/2),image.type());
        Mat gray=new Mat(image.size(),image.type());
        Mat gray0=new Mat(image.size(),CvType.CV_8U);

        // Utilizamos filtros gausianos en la imagen
        Imgproc.pyrDown(image, smallerImg, smallerImg.size());
        Imgproc.pyrUp(smallerImg, image, image.size());


        MatOfPoint approx=new MatOfPoint();
        for (int c = 0; c < 3; c++) {

            extractChannel(image, gray, c);

            // Try several threshold levels.
            for (int l = thresholdMin; l < thresholdLvl; l++) {

                int threshold = (l + 1) * thresholdMax / thresholdLvl;

                // Segmentamos la imagen con ese limite
                Imgproc.threshold(gray, gray0, threshold, thresholdMax, Imgproc.THRESH_BINARY);
                List<MatOfPoint> contours=new ArrayList<MatOfPoint>();

                // Encontramos los contornos de las imagenes
                Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                // Probamos para cada contorno
                for( int i = 0; i < contours.size(); i++ ) {

                    // Aproximamos los contornos
                    approx = approxPolyDP(contours.get(i), Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true)*0.02, true);

                    // Los contornos de los rectangulos deberian tener 4 vertices y ser convexos
                    // Como el valor puede ser positivo o negativo tomamos el valor absoluto
                    if( approx.toArray().length == 4 && Math.abs(Imgproc.contourArea(approx)) > areaSize && Imgproc.isContourConvex(approx) ) {

                        double maxCosine = 0;

                        for( int j = 2; j < 5; j++ ) {
                            // Buscamos el coseno maximo del angulo entre los vertices
                            double cosine = Math.abs(angle(approx.toArray()[j%4], approx.toArray()[j-2], approx.toArray()[j-1]));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        // Si el coseno es relativamente pequeño
                        // Lo agregamos a nuestra lista
                        if( maxCosine < 0.3 ) { squares.add(approx); }
                    }
                }

            }
        }

    }

    private void extractChannel(Mat source, Mat out, int channelNum) {
        List<Mat> sourceChannels=new ArrayList<Mat>();
        List<Mat> outChannel=new ArrayList<Mat>();

        Core.split(source, sourceChannels);

        outChannel.add(new Mat(sourceChannels.get(0).size(),sourceChannels.get(0).type()));
        Core.mixChannels(sourceChannels, outChannel, new MatOfInt(channelNum,0));
        Core.merge(outChannel, out);
    }

    private MatOfPoint approxPolyDP(MatOfPoint curve, double epsilon, boolean closed) {
        MatOfPoint2f tempMat=new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), tempMat, epsilon, closed);
        return new MatOfPoint(tempMat.toArray());
    }

}
