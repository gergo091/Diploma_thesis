#include "preprocessing.hpp"

using namespace cv;

GczOpenCV::Preprocessing::Preprocessing(const cv::Mat & image){
	this->image = image;

	image.copyTo(gray_image);
	cv::cvtColor(image, gray_image, CV_RGB2GRAY);

	horizontal_resolution = 500;
	vertical_resolution = 500;
}

GczOpenCV::Preprocessing::Preprocessing(const cv::Mat & image, int horizontal_resolution, int vertical_resolution){
	this->image = image;

	image.copyTo(gray_image);
	cv::cvtColor(image, gray_image, CV_RGB2GRAY);

	this->horizontal_resolution = horizontal_resolution;
	this->vertical_resolution = vertical_resolution;
}

cv::Mat GczOpenCV::Preprocessing::Segmentation(cv::Mat image){
	//cv::GaussianBlur(gray_image, gray_image, cv::Size(5, 5), 0, 0);

	raw_mask.create(image.rows, image.cols, gray_image.type());
	gray_image.copyTo(segmentation);

	int W = horizontal_resolution / 50;		//	size of block for segmentation
	int M;									//	average value intensity for gray block
	int W_row, W_col;						//	index in block
	int RW = W, CW = W;						//	size of last block for x and y ax
	int I;									//	intensity of gray color in point/pixel
	int V = 0;								//	variability of gray color in block

	if (W < 5)
		W = 5;

	//	traverse the whole picture
	for (int row = 0; row < image.rows; row += W){
		for (int col = 0; col < image.cols; col += W){

			M = 0;
			//	average value intensity for gray block
			for (W_row = row; W_row < row + RW; W_row++){
				for (W_col = col; W_col < col + CW; W_col++){
					M += image.at<uchar>(W_row, W_col);
				}
			}

			M /= (RW * CW);

			//	variability of gray color in block
			for (W_row = row; W_row < row + RW; W_row++){
				for (W_col = col; W_col < col + CW; W_col++){
					I = image.at<uchar>(W_row, W_col);
					V += (I - M) * (I - M);
				}
			}

			V = V / (RW * CW);

			//	comparing average intensity and average variability
			for (W_row = row; W_row < row + RW; W_row++){
				for (W_col = col; W_col < col + CW; W_col++){
					V < M ? raw_mask.at<uchar>(W_row, W_col) = 0 : raw_mask.at<uchar>(W_row, W_col) = 255;
				}
			}
			CW = supplement_cols(W, W_col);
		}
		CW = W;
		RW = supplement_rows(W, W_row);
	}

	raw_mask.copyTo(mask);


	for (int i = 0; i < raw_mask.rows; i++){
		for (int j = 0; j < raw_mask.cols; j++){
			if (mask.at<uchar>(i, j) == 0){
				mask.at<uchar>(i, j) = 255;
				continue;
			}
			if (mask.at<uchar>(i, j) == 255)
				mask.at<uchar>(i, j) = 0;
		}
	}

	wiener_to_mask(50, 50);

	for (int i = 0; i < mask.rows; i++){
		for (int j = 0; j < mask.cols; j++){
			if (mask.at<uchar>(i, j) > 120){
				mask.at<uchar>(i, j) = 255;
				continue;
			}
			if (mask.at<uchar>(i, j) <= 120)
				mask.at<uchar>(i, j) = 0;
		}
	}

	//	transfer mask to image
	for (int row = 0; row < image.rows; row++){
		for (int col = 0; col < image.cols; col++){
			if (mask.at<uchar>(row, col) == 0)
				segmentation.at<uchar>(row, col) = 255;
		}
	}

	return segmentation;
}

cv::Mat GczOpenCV::Preprocessing::OrientationMap(cv::Mat image){
	orientationMap = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
	cv::Mat gX = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
	cv::Mat gY = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
	cv::Mat gradX = cv::Mat::zeros(image.rows, image.cols, image.type());
	cv::Mat gradY = cv::Mat::zeros(image.rows, image.cols, image.type());

	//Set_sizeofBlockOrientation(31);
	int block = sizeofBlockOrientation;

	//	gradients for X and Y
	Sobel(image, gradX, CV_64FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	Sobel(image, gradY, CV_64FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	//	local orientation of pixel
	for (int k = block / 2; k < image.rows - block / 2; k++){
		for (int l = block / 2; l < image.cols - block / 2; l++){
			for (int i = k - block / 2; i < k + block / 2; i++){
				for (int j = l - block / 2; j < l + block / 2; j++){
					gY.at<double>(k, l) += 2 * (double)gradY.at<double>(i, j) * (double)gradX.at<double>(i, j);
					gX.at<double>(k, l) += pow((double)gradX.at<double>(i, j), 2) - pow((double)gradY.at<double>(i, j), 2);
				}
			}
			//	orientation for entire block
			orientationMap.at<double>(k, l) = 0.5 * atan2(gY.at<double>(k, l), gX.at<double>(k, l)) + M_PI_2;
		}
	}

	cv::Mat sin = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
	cv::Mat cos = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);


	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			sin.at<double>(i, j) = std::sin(2 * orientationMap.at<double>(i, j));
			cos.at<double>(i, j) = std::cos(2 * orientationMap.at<double>(i, j));
		}
	}

	GaussianBlur(sin, sin, cv::Size(39, 39), 10, 10);
	GaussianBlur(cos, cos, cv::Size(39, 39), 10, 10);

	for (int i = 1; i < image.rows; i++){
		for (int j = 1; j < image.cols; j++){
			orientationMap.at<double>(i - 1, j - 1) = 1 / 2.0 * (atan2((double)sin.at<double>(i, j), (double)cos.at<double>(i, j)));
		}
	}

	return orientationMap;
}

cv::Mat GczOpenCV::Preprocessing::DrawOrientationMap(cv::Mat image){
	imageOfOrientation = this->image.clone();
	int velkost_bloku = Get_sizeofBlockOrientation();

	for (int i = 0; i < image.rows / velkost_bloku; i++){
		for (int j = 0; j < image.cols / velkost_bloku; j++){
			double x1, y1, x2, y2;
			x1 = velkost_bloku / 2.0 + (j*velkost_bloku);
			y1 = velkost_bloku / 2.0 + (i*velkost_bloku);
			x2 = velkost_bloku + (j*velkost_bloku);
			y2 = velkost_bloku / 2.0 + (i*velkost_bloku);
			cv::Point bod_vypocitany(((x2 - x1)*cos(orientationMap.at<double>(i*velkost_bloku, j*velkost_bloku)) - (y2 - y1)*sin(orientationMap.at<double>(i*velkost_bloku, j*velkost_bloku))) + velkost_bloku / 2.0 + (j*velkost_bloku), ((x2 - x1)*sin(orientationMap.at<double>(i*velkost_bloku, j*velkost_bloku)) + (y2 - y1)*cos(orientationMap.at<double>(i*velkost_bloku, j*velkost_bloku))) + velkost_bloku / 2.0 + (i*velkost_bloku));
			cv::Point bod_staticky(x1, y1);
			line(imageOfOrientation, bod_staticky, bod_vypocitany, cv::Scalar(0, 0, 255), 1, 4, 0);
		}
	}
	return imageOfOrientation;
}

cv::Mat GczOpenCV::Preprocessing::GaborFilter(cv::Mat image){
	cv::Point anchor(-1, -1);
	cv::Mat kernel;
	cv::Mat tempOrientationMap = orientationMap.clone();
	gabor = cv::Mat::zeros(image.rows, image.cols, image.type());


	double delta = 0, sucet = 0.0;
	int ddepth = -1, u = 0, v = 0, count = 0;
	int kernel_size = 0;
	double psi = 0;

	/*sizeofBlockGabor = 39;
	sigma = 7.0;
	lambda = 15.0;
	gamma = 1;*/

	cv::Size dim = cv::Size(sizeofBlockGabor, sizeofBlockGabor);

	for (int i = (sizeofBlockGabor - 1) / 2; i < orientationMap.rows - (sizeofBlockGabor - 1) / 2; i++){
		for (int j = (sizeofBlockGabor - 1) / 2; j < orientationMap.cols - (sizeofBlockGabor - 1) / 2; j++){
			if (tempOrientationMap.at<double>(i, j) >M_PI_2){
				tempOrientationMap.at<double>(i, j) -= M_PI_2;
			}
			else{
				tempOrientationMap.at<double>(i, j) += M_PI_2;
			}

			kernel = cv::getGaborKernel(dim, sigma, tempOrientationMap.at<double>(i, j), lambda, gamma, psi, CV_64F);

			for (int k = i - (sizeofBlockGabor - 1) / 2; k < i + sizeofBlockGabor - (sizeofBlockGabor - 1) / 2; k++){
				for (int l = j - (sizeofBlockGabor - 1) / 2; l < j + sizeofBlockGabor - (sizeofBlockGabor - 1) / 2; l++){
					sucet += (image.at<uchar>(k, l) / 255.0) * kernel.at<double>(u, v);
					v++;
				}
				v = 0;
				u++;
			}
			u = 0;
			gabor.at<uchar>(i, j) = sucet;
			sucet = 0.0;
		}
	}

	return gabor;
}

void GczOpenCV::Preprocessing::ColourGabor(cv::Mat & image)
{
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			image.at<uchar>(i, j) = abs(image.at<uchar>(i, j) - 255.0);
		}
	}
}

//cv::Mat GczOpenCV::Preprocessing::GaborFilterOpt(cv::Mat & image){
//	Mat kernel; //maska gabora
//	cv::Mat tempOrientationMap = orientationMap.clone();
//	gabor = cv::Mat::zeros(image.rows, image.cols, image.type())
//	this->mat_filtrovana = Mat(this->mat_po_normal.rows, this->mat_po_normal.cols, CV_8U, Scalar::all(255));
//	double  sum = 0; //sigma nebola pouzivana nakolko sme si ju vedeli odvodit
//	//int u=0,v=0;
//	Mat test;
//	Mat subMat, sub;
//	int mocnina = pow(sizeofBlockGabor, 2);
//	bitwise_not(this->mat_po_normal, test);
//	int w = 0;
//	int q = 0;
//	Scalar s;
//	for (int i = (sizeofBlockGabor) / 2; i< this->mat_po_normal.rows - (velkost) / 2; i++){
//
//		if (i%velkost == 0) w++;
//
//		for (int j = (velkost) / 2; j< this->mat_po_normal.cols - (velkost) / 2; j++){
//
//			if (j%velkost == 0) q++;
//
//			if ((this->maska.at<uchar>(i - (velkost - 1) / 2, j - (velkost - 1) / 2) != 0) && (this->maska.at<uchar>(i + (velkost - 1) / 2, j + (velkost - 1) / 2) != 0)){ //ak maska na pozici nema ciernu hodnotu
//
//				kernel = getGaborKernel(Size(velkost, velkost), 10/*this->sigma.at<double>(w,q)*/, this->Theta.at<double>(w, q), 10/*this->frekvencnaMat.at<double>(w,q)*/, 0, 0, CV_64F); //vytvori sa maska Gabora
//				subMat = test(Rect(j - velkost / 2, i - velkost / 2, velkost, velkost));
//				subMat.convertTo(sub, CV_64F);
//				multiply(sub, kernel, sub);
//				s = cv::sum(sub);
//				sum = s.val[0] / mocnina;
//				this->mat_filtrovana.at<uchar>(i, j) = sum; //zapisem nove hodnoty po filtracii
//				sum = 0;
//			}
//		}
//		q = 0;
//	}
//	ui->loading->setText("");
//
//	return ;
//}

cv::Mat GczOpenCV::Preprocessing::Binarization(cv::Mat image){
	threshold(image, binarization, 100, 255, CV_THRESH_BINARY);

	return binarization;
}

cv::Mat GczOpenCV::Preprocessing::ThinningImage(cv::Mat image){
	image.copyTo(thinning_image);

	//ChangeBlacToWhite(thinning_image);

	thinning_image /= 255;

	cv::Mat prev = cv::Mat::zeros(thinning_image.size(), CV_8UC1);
	cv::Mat diff;

	do {
		GuoHall(thinning_image, 0);
		GuoHall(thinning_image, 1);
		cv::absdiff(thinning_image, prev, diff);
		thinning_image.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	thinning_image *= 255;

	//ChangeWhiteToBlack(thinning_image);
	ChangeBlacToWhite(thinning_image);

	return thinning_image;
}

/////////////////////////////////// temp methods /////////////////////////////////////

int GczOpenCV::Preprocessing::supplement_cols(int W, int W_col){
	int CW;

	if (W_col + W > image.cols)
		return CW = image.cols % W;
	else
		return CW = W;
}

int GczOpenCV::Preprocessing::supplement_rows(int W, int W_row){
	int CW;

	if (W_row + W > image.rows)
		return CW = image.rows % W;
	else
		return CW = W;
}

void GczOpenCV::Preprocessing::wiener_to_mask(int block_size_x, int block_size_y){
	cv::Mat kernel;
	cv::Mat tmp1, tmp2, tmp3, tmp4;

	int block_x = block_size_x;
	int block_y = block_size_y;

	kernel.create(block_x, block_y, CV_32F);
	kernel.setTo(cv::Scalar(1.0 / (double)(block_x * block_y)));

	tmp1.create(raw_mask.rows, raw_mask.cols, raw_mask.type());
	tmp2.create(raw_mask.rows, raw_mask.cols, raw_mask.type());
	tmp3.create(raw_mask.rows, raw_mask.cols, raw_mask.type());
	tmp4.create(raw_mask.rows, raw_mask.cols, raw_mask.type());

	cv::filter2D(raw_mask, tmp1, -1, kernel, cv::Point(block_x / 2, block_y / 2));

	cv::multiply(raw_mask, raw_mask, tmp2);
	cv::filter2D(tmp2, tmp3, -1, kernel, cv::Point(block_x / 2, block_y / 2));

	cv::multiply(tmp1, tmp1, tmp4);
	cv::subtract(tmp3, tmp4, tmp3);

	cv::Scalar noise = cv::mean(tmp3);

	cv::subtract(raw_mask, tmp1, mask);
	cv::max(tmp3, noise, tmp2);
	cv::add(tmp3, -noise, tmp3);
	cv::max(tmp3, 0, tmp3);

	cv::divide(tmp3, tmp2, tmp3);
	cv::multiply(tmp3, mask, mask);
	cv::add(mask, tmp1, mask);
}

void GczOpenCV::Preprocessing::GuoHall(cv::Mat & image, int iteration){
	cv::Mat marker = cv::Mat::zeros(image.size(), CV_8UC1);

	for (int i = 1; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{

			uchar p2 = image.at<uchar>(i - 1, j);
			uchar p3 = image.at<uchar>(i - 1, j + 1);
			uchar p4 = image.at<uchar>(i, j + 1);
			uchar p5 = image.at<uchar>(i + 1, j + 1);
			uchar p6 = image.at<uchar>(i + 1, j);
			uchar p7 = image.at<uchar>(i + 1, j - 1);
			uchar p8 = image.at<uchar>(i, j - 1);
			uchar p9 = image.at<uchar>(i - 1, j - 1);

			int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
				(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
			int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
			int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
			int N = N1 < N2 ? N1 : N2;

			int m = iteration == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if (C == 1 && (N >= 2 && N <= 3) & m == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	image &= ~marker;
}

void GczOpenCV::Preprocessing::ChangeBlacToWhite(cv::Mat & image){
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			if (image.at<uchar>(i, j) == 0){
				image.at<uchar>(i, j) = 255;
				continue;
			}
			if (image.at<uchar>(i, j) == 255)
				image.at<uchar>(i, j) = 0;
		}
	}
}

void GczOpenCV::Preprocessing::ChangeWhiteToBlack(cv::Mat & image){
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			if (image.at<uchar>(i, j) == 255){
				image.at<uchar>(i, j) = 0;
				continue;
			}
			if (image.at<uchar>(i, j) == 0)
				image.at<uchar>(i, j) = 255;
		}
	}
}
