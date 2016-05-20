#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string.h>

using namespace cv;
using namespace std;
using namespace tesseract;

Scalar color=Scalar(0,255,0);
Mat src,src_gray,adt,drawing,threshold_output,drawing_gray;

int font_thickness=1;
int font_lineType=8;
bool bottomLeftOrigin=false;
int fontFace=CV_FONT_HERSHEY_SIMPLEX;
double fontScale=0.5;

//*********Erosion Variables**************
int erosion_elem = 0;
int erosion_size = 10;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
Mat erosion_dst;
void Erosion( int, void* );
//*****************************************

//***********Morphology Operation Variables******
int morph_elem=0;
int morph_size=0;
int morph_operator=1;
int const max_operator=4;
int const max_kernel_size=21;
Mat morph_dst;
void Morphology_Operations(int, void*);
//************************************************

//***********Thresholding******************
int thresh = 100;
int max_thresh = 255;
//*****************************************

//**************Contours*******************
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
//*****************************************


//Main Function
int main( int argc, char** argv )
{	
	TessBaseAPI *api = new TessBaseAPI();
	
		
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (api->Init(NULL, "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}

	src=imread("swine.jpg");
	drawing=src.clone();
	src.copyTo(drawing);
	drawing.setTo(cv::Scalar(255,255,255));
	
	FILE * pFile;
   	pFile = fopen ("decode_text.txt","w");

	// Convert image to gray
	cvtColor(src, src_gray, CV_BGR2GRAY );
	imwrite("color converted.tif",src_gray);
	
	//Thresholding for Tesseract
	threshold(src_gray,adt, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imwrite("Adaptive_thresholding.tif",adt);

	Boxa *boxes;    	
	// Open input image with leptonica library
    	Pix *image = pixRead("Adaptive_thresholding.tif");
	api->SetImage(image);
	//boxes = api->GetComponentImages(tesseract::RIL_WORD, true, NULL, NULL);
	api->Recognize(0);
	
	//************Does the same work as ResultIterator*************
	/*for (int j= 0; j < boxes->n; j++) {
		BOX* box = boxaGetBox(boxes, j, L_CLONE);
		api->SetRectangle(box->x, box->y, box->w, box->h);
		char* ocrResult = api->GetUTF8Text();
    		int conf = api->MeanTextConf();
		int baseline=0;
		//Size textSize = getTextSize(ocrResult, fontFace,fontScale, font_thickness, &baseline);
		//baseline += font_thickness;
		Point textOrg(box->x, box->y+box->h);
		
		const char *font_name;
    		bool bold, italic, underlined, monospace, serif, smallcaps;
    		int pointsize, font_id;
    		font_name = api->WordFontAttributes(&bold, &italic, &underlined,&monospace, &serif, &smallcaps,&pointsize, &font_id);
    		fprintf(stdout, "pointsize:'%d':,font_id:'%d', confidence: %d, text: %s",pointsize,font_id, conf, ocrResult);
		rectangle( drawing, Point(box->x,box->y),Point(box->x+box->w,box->y+box->h), color, 2, 8, 0 );
		putText(drawing, ocrResult,textOrg,fontFace, fontScale, Scalar(0,0,0), font_thickness,font_lineType, bottomLeftOrigin);
		
	}*/

	
	
	tesseract::ResultIterator* ri = api->GetIterator();
	tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
	
	if (ri != 0) {
		do {
			char* ocrResult = ri->GetUTF8Text(level);
			float conf = ri->Confidence(level);
			int x1, y1, x2, y2;
      			ri->BoundingBox(level, &x1, &y1, &x2, &y2);
			double area=(x2-x1)*(y2-y1);	
			
			const char *font_name;
	    		bool bold, italic, underlined, monospace, serif, smallcaps;
	    		int pointsize, font_id;
	    		font_name = ri->WordFontAttributes(&bold, &italic, &underlined,&monospace, &serif, &smallcaps,&pointsize, &font_id);
			if(area>30.0&&pointsize<100&&strcmp(ocrResult,"")!=0&&pointsize!=0&&conf>70.0){
				rectangle( src, Point(x1,y1),Point(x2,y2), Scalar(255,255,255), -1, 8, 0 );
				rectangle( adt, Point(x1,y1),Point(x2,y2), Scalar(255,255,255), -1, 8, 0 );
				Point textOrg(x1,y2);
				//putText(src, ocrResult,textOrg,fontFace, fontScale, Scalar(0,0,0), font_thickness,font_lineType, bottomLeftOrigin);
				
			}
			
				
	      		fprintf(pFile,"word: '%s';  \tconf: %.2f;pointsize:'%d':,font_name:'%s',Area:'%lf'\n",ocrResult, conf,pointsize,font_name,area);
			
			delete[] ocrResult;
		} while (ri->Next(level));
	}
	
	imwrite("Rectangles on thresholded image.tif",adt);
	
	Morphology_Operations(0,0);

	// Convert drawing to gray
	//cvtColor(erosion_dst, drawing_gray, CV_BGR2GRAY );
	
	threshold( morph_dst, threshold_output, thresh, 255, THRESH_BINARY );
	imwrite("thresh_out.tif",threshold_output);
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector<Rect> boundRect( contours.size() );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Point2f> ContArea(contours.size());
	for( int i = 0; i < contours.size(); i++ ) {
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	}

	for( int i = 0; i< contours.size(); i++ )
     	{
	       	Scalar color = Scalar( 255,0,0 );
	    	//drawContours( src, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		//if(boundRect[i].width*boundRect[i].height>2000)
	       	//rectangle( src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
		
     	}
	imwrite("Decode_test.tif",drawing);
	imwrite("Decode_src.tif",src);
	api->End();
    	
    	pixDestroy(&image);
	fclose(pFile);
	waitKey(0);
  	return(0);
}

void Erosion( int, void* )
{
	  int erosion_type;
	  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
	  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
	  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	  Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	  // Apply the erosion operation
	  erode( adt, erosion_dst, element );

	  imwrite( "Erosion.tif", erosion_dst );
}

void Morphology_Operations(int,void*)
{
	int operation=morph_operator+2;
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ),Point( morph_size, morph_size ) );
	morphologyEx(adt,morph_dst,operation,element);
	imwrite("Morph Operation.tif",morph_dst);
}


