#ifndef preprocessing_hpp
#define preprocessing_hpp

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>	//	for cv
#include <opencv/highgui.h>		//	for draw
#include <math.h>

namespace GczOpenCV{
	class Preprocessing{
	private:
		cv::Mat image;
		cv::Mat gray_image;
		cv::Mat raw_mask;
		cv::Mat mask;
		cv::Mat segmentation;
		cv::Mat orientationMap;
		cv::Mat imageOfOrientation;
		cv::Mat gabor;
		cv::Mat binarization;
		cv::Mat thinning_image;

		int horizontal_resolution;
		int vertical_resolution;
		int sizeofBlockGabor, sizeofBlockOrientation, sigma;
		double lambda, gamma;

		int supplement_rows(int W, int W_row);
		int supplement_cols(int W, int W_col);
		void wiener_to_mask(int block_size_x, int block_size_y);
		void GuoHall(cv::Mat & image, int iteration);
		void ChangeBlacToWhite(cv::Mat & image);
		void ChangeWhiteToBlack(cv::Mat & image);

	public:
		Preprocessing();
		Preprocessing(const cv::Mat & image);
		Preprocessing(const cv::Mat & image, int horizontal_resolution, int vertical_resolution);
		cv::Mat Segmentation(cv::Mat image);        // deleted &
		cv::Mat OrientationMap(cv::Mat image);      //deleted &
		cv::Mat DrawOrientationMap(cv::Mat image);  //deleted &
		cv::Mat GaborFilter(cv::Mat image);         //deleted &
		void ColourGabor(cv::Mat & image);
		cv::Mat GaborFilterOpt(cv::Mat & image);
		cv::Mat Binarization(cv::Mat image);        //deleted &
		cv::Mat ThinningImage(cv::Mat image);       //deleted &

		//	functions
		inline cv::Mat Get_image()			{ return image; }
		inline cv::Mat Get_gray_image()		{ return gray_image; }
		inline cv::Mat Get_segmentation()	{ return segmentation; }
		inline cv::Mat Get_mask()			{ return mask; }
		inline cv::Mat Get_raw_mask()		{ return raw_mask; }
		inline cv::Mat Get_orientation()	{ return orientationMap; }
		inline cv::Mat Get_gabor_filter()	{ return gabor; }
		inline cv::Mat Get_binarization()	{ return binarization; }
		inline cv::Mat Get_thinning_image()	{ return thinning_image; }

		//	parameters
		inline int Get_sizeofBlockGabor() { return sizeofBlockGabor; }
		inline int Get_sizeofBlockOrientation() { return sizeofBlockOrientation; }
		inline int Get_sigma() { return sigma; }
		inline double Get_lambda() { return lambda; }
		inline double Get_gamma() { return gamma; }

		inline void Set_sizeofBlockGabor(int size) { sizeofBlockGabor = size; }
		inline void Set_sizeofBlockOrientation(int size) { sizeofBlockOrientation = size; }
		inline void Set_sigma(int sigma) { this->sigma = sigma; }
		inline void Set_lambda(double lambda) { this->lambda = lambda; }
		inline void Set_gamma(double gamma) { this->gamma = gamma; }
	};
}
#endif
