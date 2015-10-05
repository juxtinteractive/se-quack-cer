 /**
 * Image Filtering
 * This sketch will help us to adjust the filter values to optimize blob detection
 * 
 * Persistence algorithm by Daniel Shifmann:
 * http://shiffman.net/2011/04/26/opencv-matching-faces-over-time/
 *
 * @author: Jordi Tost (@jorditost)
 * @url: https://github.com/jorditost/ImageFiltering/tree/master/ImageFilteringWithBlobPersistence
 *
 * University of Applied Sciences Potsdam, 2014
 *
 * It requires the ControlP5 Processing library:
 * http://www.sojamo.de/libraries/controlP5/
 */
import org.opencv.core.*;
import gab.opencv.*;
import java.awt.Rectangle;
import processing.video.*;
import controlP5.*;
import oscP5.*;
import netP5.*;

OpenCV bodyCv;
OpenCV billCv;
Capture video;
PImage src, preProcessedImage, processedImage, contoursImage;

ArrayList<Contour> bodyContours;
ArrayList<Contour> billContours;

// List of detected contours parsed as blobs (every frame)
ArrayList<Contour> newBlobContours;


ArrayList<Duck> ducks;

// List of my blob objects (persistent)
ArrayList<Blob> bodyBlobList;
ArrayList<Blob> billBlobList;


// Number of blobs detected over all time. Used to set IDs.
int blobCount = 0;

float contrast = 1.35;
int brightness = 0;
int threshold = 75;

int thresholdBlockSize = 489;
int thresholdConstant = 45;
int bodyBlobSizeMinThreshold = 20;
int billBlobSizeMinThreshold = 20;
int bodyBlobSizeMaxThreshold = 100;
int billBlobSizeMaxThreshold = 100;
int blurSize = 4;
int gridXSegments = 8;
int gridYSegments = 8;



// Control vars
ControlP5 cp5;
int buttonColor;
int buttonBgColor;
controlP5.Range yellowHueRange;
controlP5.Range yellowBrightRange;
controlP5.Range redHueRange;
controlP5.Range redBrightRange;

controlP5.Range gridXRange;
controlP5.Range gridYRange;

Mat hue = null;
Mat bright = null;
Mat fMin = null;
Mat fMax = null;
Mat hF = null;
Mat bF = null;
Mat finalYellowHueBrightFilter = null;
Mat finalRedHueBrightFilter = null;
Mat frameMask = null;

OscP5 oscP5;

DisposeHandler dh;

boolean oversizedetected;

int sideBarSize = 350;
int w = 1280;
int h = 720;
void setup() {
  frameRate(15);
  
  //video = new Capture(this, 640, 480);
  
  video = new Capture(this, 1920, 1080, "USB 2.0 Camera");
  video.start();
  
  bodyCv = new OpenCV(this, w, h);
  billCv = new OpenCV(this, w, h);
  
  bodyContours = new ArrayList<Contour>();
  billContours = new ArrayList<Contour>();
  
  // Blobs list
  bodyBlobList = new ArrayList<Blob>();
  billBlobList = new ArrayList<Blob>();
  
  size(bodyCv.width + sideBarSize, bodyCv.height, P2D);
  
  // Init Controls
  cp5 = new ControlP5(this);
  initControls();
  dh = new DisposeHandler(this);
  PImage fm = loadImage("framemask.png");
  fm.resize(w,h);
  OpenCV t = new OpenCV(this,w,h);
  t.loadImage(fm);
  frameMask =  t.getGray();
  
  oscP5 = new OscP5(this,"127.0.0.1",1235);
}

void draw() {
  
  // Read last captured frame
  if (video.available()) {
    video.read();
  }
  
  // Load the new frame of our camera in to OpenCV
  PImage img = createImage(video.width,video.height,RGB);
  img.copy(video,0,0,video.width,video.height,0,0,video.width,video.height);
  if(frameCount == 10){
    img.save("earlyframe.png");
  }
  img.resize(w,h);
  
  
  bodyCv.loadImage(img);
  src = bodyCv.getSnapshot();
  
  ///////////////////////////////
  // <1> PRE-PROCESS IMAGE
  // - Grey channel 
  // - Brightness / Contrast
  ///////////////////////////////
  
  // Gray channel
 
  bodyCv.useColor(HSB);
  
  hue = bodyCv.getH();
  bright = bodyCv.getB();
  
  if(fMin == null){ fMin = hue.clone(); }
  if(fMax == null){ fMax = hue.clone(); }
  if(hF == null){ hF = hue.clone(); }
  if(bF == null){ bF = hue.clone(); } 
  if(finalYellowHueBrightFilter == null){ finalYellowHueBrightFilter = hue.clone(); }
  if(finalRedHueBrightFilter == null){ finalRedHueBrightFilter = hue.clone(); }

  
  
  org.opencv.imgproc.Imgproc.threshold(hue,fMin,yellowHueRange.getLowValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY);
  org.opencv.imgproc.Imgproc.threshold(hue,fMax,yellowHueRange.getHighValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY_INV);
  org.opencv.core.Core.bitwise_and(fMin,fMax,hF);
  org.opencv.core.Core.bitwise_and(hF,frameMask,hF);
  
  org.opencv.imgproc.Imgproc.threshold(bright,fMin,yellowBrightRange.getLowValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY);
  org.opencv.imgproc.Imgproc.threshold(bright,fMax,yellowBrightRange.getHighValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY_INV);
  org.opencv.core.Core.bitwise_and(fMin,fMax,bF);
  org.opencv.core.Core.bitwise_and(bF,frameMask,bF);
  
  org.opencv.core.Core.bitwise_and(bF,hF,finalYellowHueBrightFilter);
  
  org.opencv.imgproc.Imgproc.threshold(hue,fMin,redHueRange.getLowValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY);
  org.opencv.imgproc.Imgproc.threshold(hue,fMax,redHueRange.getHighValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY_INV);
  org.opencv.core.Core.bitwise_and(fMin,fMax,hF);
  org.opencv.core.Core.bitwise_and(hF,frameMask,hF);
  
  org.opencv.imgproc.Imgproc.threshold(bright,fMin,redBrightRange.getLowValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY);
  org.opencv.imgproc.Imgproc.threshold(bright,fMax,redBrightRange.getHighValue(),255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY_INV);
  org.opencv.core.Core.bitwise_and(fMin,fMax,bF);
  org.opencv.core.Core.bitwise_and(bF,frameMask,bF);
  
  org.opencv.core.Core.bitwise_and(bF,hF,finalRedHueBrightFilter);
  
  bodyCv.setGray(finalYellowHueBrightFilter);
  bodyCv.dilate();
  bodyCv.erode();
  bodyCv.blur(blurSize);
  src = bodyCv.getSnapshot();
  
  billCv.setGray(finalRedHueBrightFilter);
  billCv.dilate();
  billCv.erode();
  billCv.blur(blurSize);
  preProcessedImage = billCv.getSnapshot();
//  opencv.invert();
  
  
  processedImage = billCv.getSnapshot();

  // Invert (black bg, white blobs)
  
  ///////////////////////////////
  // <3> FIND CONTOURS  
  ///////////////////////////////
  
  //bodies
  detectBlobs(false);
  //bills
  detectBlobs(true);
  //ducks
  buildDucks();
    
  // Save snapshot for display
//  contoursImage = opencv.getSnapshot();
  
  // Draw
  pushMatrix();
    
    // Leave space for ControlP5 sliders
    translate(width-src.width, 0);
    
    // Display images
    displayImages();
    
    // Display contours in the lower right window
    pushMatrix();
      scale(0.5);
      translate(src.width, src.height);
      
      // Contours
      //displayContours();
      //displayContoursBoundingBoxes();
      
      // Blobs
      drawGrid();
//      displayBlobs();
      drawDucks();
      
    popMatrix(); 
    
  popMatrix();
  sendOSC();
}

///////////////////////
// Display Functions
///////////////////////

void displayImages() {
  
  pushMatrix();
  scale(0.5);
  image(src, 0, 0);
  image(preProcessedImage, src.width, 0);
  image(processedImage, 0, src.height);
  image(src, src.width, src.height);
  popMatrix();
  
  stroke(255);
  fill(255);
  textSize(12);
  text("Source", 10, 25); 
  text("Pre-processed Image", src.width/2 + 10, 25); 
  text("Processed Image", 10, src.height/2 + 25); 
  text("Tracked Points", src.width/2 + 10, src.height/2 + 25);
}

void displayBlobs() {
  
  for (Blob b : bodyBlobList) {
    strokeWeight(1);
    b.display();
  }
  for (Blob b : billBlobList) {
    strokeWeight(1);
    b.display();
  }
}

void drawDucks(){
 for(Duck d : ducks){
   d.body.display();
   d.bill.display();
  PVector dc = d.body.getCenter();
  PVector ic = d.bill.getCenter();
  stroke(0,255,0);
  line(dc.x,dc.y,ic.x,ic.y);
 } 
}
void drawGrid(){
 float xs = gridXRange.getLowValue();
 float xe = gridXRange.getHighValue();
 float xd = xe-xs;
 float xi = xd / gridXSegments;
 float ys = gridYRange.getLowValue();
 float ye = gridYRange.getHighValue();
 float yd = ye-ys;
 float yi = yd / gridYSegments;
 stroke(255);
 for(int x = 0; x <= gridXSegments; x++){
   line(xs + xi *x,ys,xs + xi *x,ye);
 }
 for(int y = 0; y <= gridYSegments; y++){
   line(xs,ys + yi * y,xe,ys + yi * y);
 }
}
//void displayContours() {
//  
//  // Contours
//  for (int i=0; i<contours.size(); i++) {
//  
//    Contour contour = contours.get(i);
//    
//    noFill();
//    stroke(0, 255, 0);
//    strokeWeight(3);
//    contour.draw();
//  }
//}

//void displayContoursBoundingBoxes() {
//  
//  for (int i=0; i<contours.size(); i++) {
//    
//    Contour contour = contours.get(i);
//    Rectangle r = contour.getBoundingBox();
//    
//    if (//(contour.area() > 0.9 * src.width * src.height) ||
//        (r.width < bodyBlobSizeMinThreshold || r.height < bodyBlobSizeMinThreshold))
//      continue;
//    
//    stroke(255, 0, 0);
//    fill(255, 0, 0, 150);
//    strokeWeight(2);
//    rect(r.x, r.y, r.width, r.height);
//  }
//}

////////////////////
// Blob Detection
////////////////////

void detectBlobs(boolean bills) {
  
  
  // Contours detected in this frame
  // Passing 'true' sorts them by descending area.
  ArrayList<Contour> contours = null;
  ArrayList<Blob> blobList = null;
  if(bills){
    billContours = billCv.findContours(true, true);
    contours = billContours;
    blobList = billBlobList;
  }else{
    bodyContours = bodyCv.findContours(true, true);
    contours = bodyContours;
    blobList = bodyBlobList;
  }
  
  newBlobContours = filterContours(contours,bills);
  
  //println(contours.length);
  
  // Check if the detected blobs already exist are new or some has disappeared. 
  
  // SCENARIO 1 
  // blobList is empty
  if (blobList.isEmpty()) {
    // Just make a Blob object for every face Rectangle
    for (int i = 0; i < newBlobContours.size(); i++) {
      println("+++ New blob detected with ID: " + blobCount);
      blobList.add(new Blob(this, blobCount, newBlobContours.get(i),bills));
      blobCount++;
    }
  
  // SCENARIO 2 
  // We have fewer Blob objects than face Rectangles found from OpenCV in this frame
  } else if (blobList.size() <= newBlobContours.size()) {
    boolean[] used = new boolean[newBlobContours.size()];
    // Match existing Blob objects with a Rectangle
    for (Blob b : blobList) {
       // Find the new blob newBlobContours.get(index) that is closest to blob b
       // set used[index] to true so that it can't be used twice
       float record = 50000;
       int index = -1;
       for (int i = 0; i < newBlobContours.size(); i++) {
         float d = dist(newBlobContours.get(i).getBoundingBox().x, newBlobContours.get(i).getBoundingBox().y, b.getBoundingBox().x, b.getBoundingBox().y);
         //float d = dist(blobs[i].x, blobs[i].y, b.r.x, b.r.y);
         if (d < record && !used[i]) {
           record = d;
           index = i;
         } 
       }
       // Update Blob object location
       used[index] = true;
       b.update(newBlobContours.get(index));
    }
    // Add any unused blobs
    for (int i = 0; i < newBlobContours.size(); i++) {
      if (!used[i]) {
        println("+++ New blob detected with ID: " + blobCount);
        blobList.add(new Blob(this, blobCount, newBlobContours.get(i),bills));
        //blobList.add(new Blob(blobCount, blobs[i].x, blobs[i].y, blobs[i].width, blobs[i].height));
        blobCount++;
      }
    }
  
  // SCENARIO 3 
  // We have more Blob objects than blob Rectangles found from OpenCV in this frame
  } else {
    // All Blob objects start out as available
    for (Blob b : blobList) {
      b.available = true;
    } 
    // Match Rectangle with a Blob object
    for (int i = 0; i < newBlobContours.size(); i++) {
      // Find blob object closest to the newBlobContours.get(i) Contour
      // set available to false
       float record = 50000;
       int index = -1;
       for (int j = 0; j < blobList.size(); j++) {
         Blob b = blobList.get(j);
         float d = dist(newBlobContours.get(i).getBoundingBox().x, newBlobContours.get(i).getBoundingBox().y, b.getBoundingBox().x, b.getBoundingBox().y);
         //float d = dist(blobs[i].x, blobs[i].y, b.r.x, b.r.y);
         if (d < record && b.available) {
           record = d;
           index = j;
         } 
       }
       // Update Blob object location
       Blob b = blobList.get(index);
       b.available = false;
       b.update(newBlobContours.get(i));
    } 
    // Start to kill any left over Blob objects
    for (Blob b : blobList) {
      if (b.available) {
        b.countDown();
        if (b.dead()) {
          b.delete = true;
        } 
      }
    } 
  }
  
  // Delete any blob that should be deleted
  for (int i = blobList.size()-1; i >= 0; i--) {
    Blob b = blobList.get(i);
    if (b.delete) {
      blobList.remove(i);
    } 
  }
}

ArrayList<Contour> filterContours(ArrayList<Contour> newContours, boolean bills) {
  
  ArrayList<Contour> newBlobs = new ArrayList<Contour>();
  
  // Which of these contours are blobs?
  for (int i=0; i<newContours.size(); i++) {
    
    Contour contour = newContours.get(i);
    Rectangle r = contour.getBoundingBox();
    float area = r.width * r.height;
    if(bills){
      
      if (( area < billBlobSizeMinThreshold * billBlobSizeMinThreshold)) continue;
      if (( area > billBlobSizeMaxThreshold * billBlobSizeMaxThreshold)) continue;
      }
    else{
      if ((area < bodyBlobSizeMinThreshold*bodyBlobSizeMinThreshold)) continue;
      if ((area > bodyBlobSizeMaxThreshold*bodyBlobSizeMaxThreshold)){
        oversizedetected = true;
        continue;
      }
    }
    
    newBlobs.add(contour);
  }
  
  return newBlobs;
}

void buildDucks(){
  ducks = new ArrayList<Duck>();
   float xs = gridXRange.getLowValue();
   float xe = gridXRange.getHighValue();
   float xd = xe-xs;
   float xi = xd / gridXSegments;
   float ys = gridYRange.getLowValue();
   float ye = gridYRange.getHighValue();
   float yd = ye-ys;
   float yi = yd / gridYSegments; 
  boolean[] claimedBills = new boolean[billBlobList.size()];
  for( int i = 0; i < bodyBlobList.size(); i++){
    Blob t = bodyBlobList.get(i);
    PVector tc = t.getCenter();
    float record = 500000;
    int index = -1;
    for (int j = 0; j < billBlobList.size(); j++) {
       Blob b = billBlobList.get(j);
       float d = tc.dist(b.getCenter());
       //float d = dist(blobs[i].x, blobs[i].y, b.r.x, b.r.y);
       if (t.getBoundingBox().intersects(b.getBoundingBox()) && !claimedBills[j]) {
         
         index = j;
       }
    }
   if(index != -1 ){
     
     
     
     stroke(255);
     int x = 0;
     float tx = t.getCenter().x;
     float ty = t.getCenter().y;
     for(; x <= gridXSegments; x++){
       if(xs + xi *x > tx){
        x -= 1;
        break; 
       }
     }
     int y = 0;
     for(; y <= gridYSegments; y++){
       if(ys + yi * y > ty){
        y -= 1;
       break; 
       }
     }
     if(x > -1 && x < gridXSegments && y > -1 && y < gridYSegments){
       claimedBills[index] = true;
       Duck d = new Duck(t,billBlobList.get(index));
       d.locX = x;
       d.locY = y;
       ducks.add(d);
     }
   }
   
  }
}

Blob getBodyBlobById(int id){
  for( Blob b : bodyBlobList){
   if(b.id == id){ return b; }
  }
  return null;
}

Blob getBillBlobById(int id){
  for( Blob b : billBlobList){
   if(b.id == id){ return b; }
  }
  return null;
}

Duck getDuckByXY(int x, int y){
  for( Duck d : ducks){
   if(d.locX == x && d.locY == y){ return d; }
  }
  return null;
}

void sendOSC(){
  if(oversizedetected){
   oversizedetected = false;
   return; 
  }
  for(int x = 0; x < gridXSegments; x++){
    for(int y = 0; y < gridYSegments; y++){
      Duck d = getDuckByXY(x,y);
      float val = -1;
      if(d != null){
       val = d.getAngle(); 
      }
      String oscStr = String.format("/duck/%d/%d",x,y);
      println(String.format("%s %f",oscStr,val));
      OscMessage myOscMessage = new OscMessage(oscStr);
      
      /* add a value (an integer) to the OscMessage */
      myOscMessage.add(val);
      
      /* send the OscMessage to the multicast group. 
       * the multicast group netAddress is the default netAddress, therefore
       * you dont need to specify a NetAddress to send the osc message.
       */
      oscP5.send(myOscMessage); 
    }
  }
}

//////////////////////////
// CONTROL P5 Functions
//////////////////////////

void initControls() {
  int sh = h;
  int controlWidth = 200;
  int h = 50;
  int hSpace = 15;
  // Yellow Hue Range
  yellowHueRange = cp5.addRange("yellowHue")
     .setLabel("Yellow Hue")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,255.0)
     .setRangeValues(128.0,200.0)
     ;
     h += hSpace;
  // Yellow Brightness Range
  yellowBrightRange = cp5.addRange("yellowBright")
     .setLabel("Yellow Bright")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,255.0)
     .setRangeValues(128.0,200.0)
     ;
     h += hSpace;
  // Red Hue Range
  redHueRange = cp5.addRange("redHue")
     .setLabel("Red Hue")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,255.0)
     .setRangeValues(128.0,200.0)
     ;
     h += hSpace;
  // Red Brightness Range
  redBrightRange = cp5.addRange("redBright")
     .setLabel("Red Bright")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,255.0)
     .setRangeValues(128.0,200.0)
     ;
     h += hSpace;
     h += hSpace;
  // Slider for contrast
  cp5.addSlider("contrast")
     .setLabel("contrast")
     .setPosition(20,h)
     .setRange(0.0,6.0)
     ;
     h += hSpace;
  // Slider for threshold
  cp5.addSlider("threshold")
     .setLabel("threshold")
     .setPosition(20,h)
     .setRange(0,255)
     ;
    h += hSpace;
  // Slider for blur size
  cp5.addSlider("blurSize")
     .setLabel("blur size")
     .setPosition(20,h)
     .setRange(1,20)
     ;
     h += hSpace;
     h += hSpace;
  // Slider for minimum blob size
  cp5.addSlider("bodyBlobSizeMinThreshold")
     .setLabel("min body blob size")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0,60)
     ;
     h += hSpace;
  
  // Slider for minimum blob size
  cp5.addSlider("billBlobSizeMinThreshold")
     .setLabel("min bill blob size")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0,60)
     ;
     h += hSpace;
  // Slider for max blob size
  cp5.addSlider("bodyBlobSizeMaxThreshold")
     .setLabel("max body blob size")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(10,300)
     ;
     h += hSpace;
  
  // Slider for minimum blob size
  cp5.addSlider("billBlobSizeMaxThreshold")
     .setLabel("max bill blob size")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(10,300)
     ;
     h += hSpace;   
  
  // Slider for minimum blob size
  cp5.addSlider("gridXSegments")
     .setLabel("Grid X Segments")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(1,32)
     ;
     h += hSpace;
   cp5.addSlider("gridYSegments")
     .setLabel("Grid Y Segments")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(1,32)
     ;
     h += hSpace;
     
   gridXRange = cp5.addRange("xRange")
     .setLabel("x Range")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,w)
     .setRangeValues(0,w)
     ;
     h += hSpace;
   gridYRange = cp5.addRange("yRange")
     .setLabel("Y Range")
     .setWidth(controlWidth)
     .setPosition(20,h)
     .setRange(0.0,sh)
     .setRangeValues(0,sh)
     ;
     h += hSpace;
     
     
     
  // Store the default background color, we gonna need it later
  buttonColor = cp5.getController("contrast").getColor().getForeground();
  buttonBgColor = cp5.getController("contrast").getColor().getBackground();
  cp5.loadProperties(("props.properties"));
}


public class DisposeHandler
{
  DisposeHandler(PApplet pa)
  {
    pa.registerDispose(this);
  }
 
  public void dispose()
  {
    print("saving props");
    cp5.saveProperties(("props.properties"));
  }
}




