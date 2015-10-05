class Duck{
  public Blob body;
  public Blob bill;
  public int locX;
  public int locY;
  Duck(Blob body, Blob bill){
   this.body = body; 
   this.bill = bill;  
  }
  public float getAngle(){
   PVector bodyCenter = this.body.getCenter();
   PVector billCenter = this.bill.getCenter();
   return PVector.angleBetween(bodyCenter, billCenter);
  }
}
