#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace cv;
using namespace std;
void Erosion( int, void* );

Mat white;
cv::Mat getGrayCDF(cv::Mat, int);
cv::Mat image;
Mat drawing ; 
Mat original;
Mat drawing2;

Mat src;
Mat src_gray;
Mat draw_gray;


//void cornerHarris_demo( int, void* );
void thresh_callback(int, void* );
void Morphology_Operations( int, void* );

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
int morph_size=10;
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

int main(int argc, char** argv )
{
	image=imread("Decode_src.tif");
	original=imread("swine.jpg");
	drawing=image.clone();
	image.copyTo(drawing);
	drawing.setTo(cv::Scalar(255,255,255));
	drawing2=image.clone();
	image.copyTo(drawing2);
	
	int bins = 256;
	cv::Mat cdf = getGrayCDF(image,bins);
	cv::Mat diffy = cdf>0.7;
	cv::Mat NonZero_Locations;   // output, locations of non-zero pixels 
	cv::findNonZero(diffy, NonZero_Locations);
	double highThreshold = double((NonZero_Locations.at<cv::Point>(0).y))/bins;
	double lowThreshold = 0.2*highThreshold;
	cv::Mat contours;

	// cv::GaussianBlur( image, contours, cv::Size(7,7),2 ); // NOT REQUIRED HERE. Synthetic image
	cv::Canny( image, contours, lowThreshold*bins, highThreshold*bins);

	
	std::vector<cv::Vec4i> lines;
	double rho = 1; // delta_rho resolution
	double theta = CV_PI/180; // delta_theta resolution
	int threshold = 50; // threshold number of votes , I SET A HIGH VALUE TO FIND ONLY THE LONG LINES
	double minLineLength = 20; // min length for a line
	double maxLineGap = 2; // max allowed gap along the line
	  /// Convert image to gray and blur it
	  

	//white=imread("white.jpg");
	cv::HoughLinesP(contours,lines, rho, theta, threshold, minLineLength, maxLineGap); // running probabilistic hough line

	if (image.channels()!=3) {
		cv::cvtColor(image,image,CV_GRAY2BGR);
	} // so we can see the red lines

	int line_thickness = 2;
	cv::Scalar color=cv::Scalar(0,0,255);
	std::vector<cv::Vec4i>::const_iterator iterator_lines = lines.begin();
	while (iterator_lines!=lines.end()) {
	    cv::Point pt1((*iterator_lines)[0],(*iterator_lines)[1]);
	    cv::Point pt2((*iterator_lines)[2],(*iterator_lines)[3]);
	    cv::line( drawing, pt1, pt2, color, line_thickness);
	    ++iterator_lines;
	}
	 
	imwrite("Drawing.tif",drawing);
	cout<<drawing.size();
	cvtColor( drawing, src_gray, CV_BGR2GRAY );
	
	//Erosion( 0, 0 );

	//blur( src_gray, src_gray, Size(3,3) );
	thresh_callback( 0, 0 );
	cvtColor( drawing2, src_gray, CV_BGR2GRAY );
	Morphology_Operations( 0,0 );

	//cornerHarris_demo( 0,0 );
	//imwrite("table_removed.tif",drawing2);
	//waitKey(0);
	//cv::destroyWindow("found lines");
	return 0;
}

cv::Mat getGrayCDF(cv::Mat Input, int histSize){
	cv::Mat InputGray = Input.clone();
	if (InputGray.channels()!=1)
	{	
		cv::cvtColor(Input,InputGray,CV_BGR2GRAY);
	}
	float range[] = { 0, histSize  } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat hist;
	calcHist( &InputGray, 1, 0, cv::Mat(), hist, 1, &histSize , &histRange, uniform, accumulate );
	
	for (int i = 1; i < hist.rows; i++)
	{
		float* data = hist.ptr<float>(0);
		data[i] += data[i-1];
	}
	return hist/(InputGray.total()); // WE NOW HAVE A *NORMALIZED* COMPUTED CDF!
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Detect edges using Threshold
	threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

	// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	int hash[10000]={0};
	for( int i = 0; i < contours.size(); i++ )
//hierarchy[i][3] is the parent of ith contour
//hierarchy[hierarchy[i][3]][3] is the grandparent of ith contour

	{	if((hierarchy[hierarchy[i][3]][3])!=-1 && hierarchy[(hierarchy[hierarchy[i][3]][3]))][3]==-1)
			hash[hierarchy[i][3]]++;
	
	}
	for( int i = 0; i < contours.size(); i++ )
	{
		cout<<hash[i]<<" "<<endl;
	
	}
	
	// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	cout<<"no.of contours "<<contours.size()<<" ";
	for( int i = 0; i < contours.size(); i++ )
	{
		cout<<hierarchy[i][3]<<" ";
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect( Mat(contours_poly[i]));
	      
	}

	//imwrite("found lines.tif", drawing); 
	// Draw polygonal contour + bonding rects + circles
	  
	for( int i = 0; i< contours.size(); i++)
	{ 
		if((boundRect[i].width*boundRect[i].height)>1000&&hash[i]>4)
		//if((boundRect[i].width*boundRect[i].height)>1000)
			{
				rectangle(original, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );
		
			}
	       
	      
	}


	  /// Show in a window
	  //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imwrite("bounding_on_table.tif",original);

	//waitKey(0);
	//cv::destroyWindow("found lines");

}

/*void cornerHarris_demo( int, void* )
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros( drawing.size(), CV_32FC1 );
	// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	int num_corners=0;
	// Detecting corners
	cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	// Normalizing
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );

	// Drawing a circle around corners
	for( int j = 0; j < dst_norm.rows ; j++ )
	{
		for( int i = 0; i < dst_norm.cols; i++ )
			{
				if( (int) dst_norm.at<float>(j,i) > thresh )
					{
						circle( dst_norm_scaled, Point( i, j ), 1,  Scalar(0,0,255), 2, 8, 0 );
						num_corners++;
		      			}
		  	}
	}
	// Showing the result
	//namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
	imwrite("CornerDetect.jpg",dst_norm_scaled);
	cout<<"No.of corners"<<num_corners<<endl;
}
*/

void Erosion( int, void* )
{
	  int erosion_type;
	  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
	  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
	  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	  Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
	  // Apply the erosion operation
	  erode( src_gray, erosion_dst, element );

	  //imwrite( "Erosion.tif", erosion_dst );
}

void Morphology_Operations(int,void*)
{
	int operation=morph_operator+2;
	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ),Point( morph_size, morph_size ) );
	morphologyEx(src_gray,src_gray,operation,element);
	imwrite("table_removed.tif",src_gray);
}


