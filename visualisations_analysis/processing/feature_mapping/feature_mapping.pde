int ctr=0;
int ctr_global = 0;
Table inputs;
Table outputs;
PImage map;
PImage digger;
PImage background;
PImage background_2;


void setup(){
  size(1100, 500);
  inputs = loadTable("/Volumes/GoogleDrive/My Drive/nerf-pytorch-master/visualisations/data/box_inputs.csv");
  map = loadImage("/Volumes/GoogleDrive/My Drive/nerf-pytorch-master/visualisations/data/map.png");
  digger = loadImage("/Volumes/GoogleDrive/My Drive/nerf-pytorch-master/visualisations/data/digger_4.png");
  background = loadImage("ref.png");
  background_2 = loadImage("ref_2.png");
  outputs = loadTable("/Volumes/GoogleDrive/My Drive/nerf-pytorch-master/visualisations/data/box_outputs.csv");
  frameRate(10);
}

void draw(){
  background(255);
  if (ctr_global == ctr){
  image(background,0,0,1100,500);
  }else{
  image(background_2,0,0,1100,500);
    
  }
  
  //Draw Box
  
  int N = inputs.getRowCount();
  int box_origin_x = 100;
  int box_origin_y = 375;
  int l = 200;
  int p=50;
  int digger_w = 250;
  
  
  stroke(0);
  strokeWeight(1);
  
  //Draw cube connecting lines
  line(box_origin_x,box_origin_y, box_origin_x+p,box_origin_y-p);
  line(box_origin_x+l,box_origin_y, box_origin_x+l+p,box_origin_y-p);
  line(box_origin_x,box_origin_y-l, box_origin_x+p,box_origin_y-l-p);
  line(box_origin_x+l,box_origin_y-l, box_origin_x+l+p,box_origin_y-l-p);
  
  //Draw back of cube
  line(box_origin_x+p, box_origin_y-p, box_origin_x+l+p,box_origin_y-p);
  line(box_origin_x+p, box_origin_y-p, box_origin_x+p,box_origin_y-l-p);
  line(box_origin_x+l+p, box_origin_y-p, box_origin_x+l+p,box_origin_y-l-p);
  line(box_origin_x+p, box_origin_y-l-p, box_origin_x+l+p,box_origin_y-l-p);
  
  //Draw Circles on Digger (Front)
  noStroke();
  for (int i = 0; i < 10; i = i+1) {
    int ctr_ref = (ctr - i)%N;
    if (ctr_ref < 0){
      ctr_ref = ctr_ref+N;
    }
    
    float draw_x = (inputs.getRow(ctr_ref).getFloat(0)/2.0+0.5)*l+(inputs.getRow(ctr_ref).getFloat(1)/2.0+0.5)*p+box_origin_x;
    float draw_y = -(inputs.getRow(ctr_ref).getFloat(2)/2.0+0.5)*l-(inputs.getRow(ctr_ref).getFloat(1)/2.0+0.5)*p+box_origin_y;
    fill(178, 91, 245, 255*(10-i)/10);
    if (inputs.getRow(ctr_ref).getFloat(1) > 0){
      ellipse(draw_x,draw_y,10,10);
    }
  }
  
  //Draw Digger
  image(digger, box_origin_x, box_origin_y-digger_w,digger_w,digger_w);
  
  //Draw Circles on Digger (Front)
  noStroke();
  for (int i = 0; i < 10; i = i+1) {
    int ctr_ref = (ctr - i)%N;
    if (ctr_ref < 0){
      ctr_ref = ctr_ref+N;
    }
    
    float draw_x = (inputs.getRow(ctr_ref).getFloat(0)/2.0+0.5)*l+(inputs.getRow(ctr_ref).getFloat(1)/2.0+0.5)*p+box_origin_x;
    float draw_y = -(inputs.getRow(ctr_ref).getFloat(2)/2.0+0.5)*l-(inputs.getRow(ctr_ref).getFloat(1)/2.0+0.5)*p+box_origin_y;
    fill(178, 91, 245, 255*(10-i)/10);
    if (inputs.getRow(ctr_ref).getFloat(1) < 0){
      ellipse(draw_x,draw_y,10,10);
    }
  }
  stroke(0);
  strokeWeight(1);
  
  //Draw front face of cube
  line(box_origin_x, box_origin_y, box_origin_x+l,box_origin_y);
  line(box_origin_x, box_origin_y, box_origin_x,box_origin_y-l);
  line(box_origin_x+l, box_origin_y, box_origin_x+l,box_origin_y-l);
  line(box_origin_x, box_origin_y-l, box_origin_x+l,box_origin_y-l);
  
  //Now the feature map (output)
  int map_origin_x = 700;
  int map_origin_y = 400;
  int w = 300;
  int n_gridlines = 10;
  
  
  //1. Draw Image
  if (ctr_global != ctr){
    image(map, map_origin_x, map_origin_y-w, w, w);
  }
  
  
  //2. Draw Grid Lines for Feature Map
  strokeWeight(1);
  stroke(0,0,0,70);
  for (int i = 0; i < n_gridlines; i =i+1){
    line(map_origin_x,map_origin_y - w*i/n_gridlines,map_origin_x+w, map_origin_y - w*i/n_gridlines);
  }
  for (int i = 0; i < n_gridlines; i =i+1){
    line(map_origin_x + w*i/n_gridlines,map_origin_y,map_origin_x+w*i/n_gridlines, map_origin_y - w);
  }
  
  strokeWeight(1);
  stroke(0);
  noFill();
  rect(map_origin_x, map_origin_y-w, w, w);
  
  //2. Draw Point on Image
  noStroke();
  for (int i = 0; i < 10; i = i+1) {
    int ctr_ref = (ctr - i)%N;
    if (ctr_ref < 0){
      ctr_ref = ctr_ref+N;
    }
    float draw_x = map_origin_x + (outputs.getRow(ctr_ref).getFloat(0)/2+0.5)*w;
    float draw_y = map_origin_y - w + (outputs.getRow(ctr_ref).getFloat(1)/2+0.5)*w;
    fill(178, 91, 245, 255*(10-i)/10);
    ellipse(draw_x,draw_y,10,10);
  }
  save("anim/anim_img"+str(ctr_global)+".png");
  
  ctr = ctr+1;
  ctr = ctr%N;
  ctr_global = ctr_global+1;
  ctr_global =ctr_global%(2*N);
}
