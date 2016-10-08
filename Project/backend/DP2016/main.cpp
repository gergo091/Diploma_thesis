//#include <fann.h>
#include <doublefann.h>
#include <vector>
#include <iostream>
#include <fstream>
//#include <io.h>
#include <opencv2/highgui/highgui.hpp>
#include <cctype>
#include <map>
#include <set>
#include <opencv/cv.h>
#include <cstdio>
#include "preprocessing.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace GczOpenCV;

Mat ending, input, output_cn;
Mat bifuraction, item;
vector<int> hidLayers;
vector <vector<int> > positions,detected_minutiae;

void learn_NN(int volba, int num_input, int num_output, int num_layers, unsigned int num_neurons_hidden1, float desired_error, unsigned int max_epochs, int act_function_hidden, int act_function_output){

	Mat test_dat;
	String file;
	string train_files;
	struct fann *ann = fann_create_standard(3, 3, 15, 2);

	//const float desired_error = (const float) 0.0001;
	//const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 100;

	if (volba == 1){
	    cout << "\nTrain NN for Basic minutiae" << endl;
		/*num_input = 9;
		num_output = 2;
		num_layers = 3;
		const unsigned int num_neurons_hidden1 = 15;*/
		ann = fann_create_standard(num_layers, num_input, num_neurons_hidden1, num_output);
	}
	else if (volba == 2){
	    cout << "\nTrain NN for Complex minutiae" << endl;
		/*num_input = 51 * 51;
		num_output = 3;
		num_layers = 3;
		const unsigned int num_neurons_hidden2 = 3500;*/
		ann = fann_create_standard(num_layers, num_input, num_neurons_hidden1, num_output);
	}

    switch (act_function_hidden){
        case 0:
            fann_set_activation_function_hidden(ann, FANN_LINEAR);
            break;
        case 1:
            fann_set_activation_function_hidden(ann, FANN_THRESHOLD);
            break;
        case 2:
            fann_set_activation_function_hidden(ann, FANN_THRESHOLD_SYMMETRIC);
            break;
        case 3:
            fann_set_activation_function_hidden(ann, FANN_SIGMOID);
            break;
        case 4:
            fann_set_activation_function_hidden(ann, FANN_SIGMOID_STEPWISE);
            break;
        case 5:
            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
            break;
        case 6:
            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
            break;
        case 7:
            fann_set_activation_function_hidden(ann, FANN_GAUSSIAN);
            break;
        case 8:
            fann_set_activation_function_hidden(ann, FANN_GAUSSIAN_SYMMETRIC);
            break;
        case 9:
            fann_set_activation_function_hidden(ann, FANN_GAUSSIAN_STEPWISE);
            break;
        case 10:
            fann_set_activation_function_hidden(ann, FANN_ELLIOT);
            break;
        case 11:
            fann_set_activation_function_hidden(ann, FANN_ELLIOT_SYMMETRIC);
            break;
        case 12:
            fann_set_activation_function_hidden(ann, FANN_LINEAR_PIECE);
            break;
        case 13:
            fann_set_activation_function_hidden(ann, FANN_LINEAR_PIECE_SYMMETRIC);
            break;
        case 14:
            fann_set_activation_function_hidden(ann, FANN_SIN_SYMMETRIC);
            break;
        case 15:
            fann_set_activation_function_hidden(ann, FANN_COS_SYMMETRIC);
            break;
        case 16:
            fann_set_activation_function_hidden(ann, FANN_SIN);
            break;
        case 17:
            fann_set_activation_function_hidden(ann, FANN_COS);
            break;
        default:
            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
            break;

    }

        switch (act_function_output){
        case 0:
            fann_set_activation_function_output(ann, FANN_LINEAR);
            break;
        case 1:
            fann_set_activation_function_output(ann, FANN_THRESHOLD);
            break;
        case 2:
            fann_set_activation_function_output(ann, FANN_THRESHOLD_SYMMETRIC);
            break;
        case 3:
            fann_set_activation_function_output(ann, FANN_SIGMOID);
            break;
        case 4:
            fann_set_activation_function_output(ann, FANN_SIGMOID_STEPWISE);
            break;
        case 5:
            fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
            break;
        case 6:
            fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
            break;
        case 7:
            fann_set_activation_function_output(ann, FANN_GAUSSIAN);
            break;
        case 8:
            fann_set_activation_function_output(ann, FANN_GAUSSIAN_SYMMETRIC);
            break;
        case 9:
            fann_set_activation_function_output(ann, FANN_GAUSSIAN_STEPWISE);
            break;
        case 10:
            fann_set_activation_function_output(ann, FANN_ELLIOT);
            break;
        case 11:
            fann_set_activation_function_output(ann, FANN_ELLIOT_SYMMETRIC);
            break;
        case 12:
            fann_set_activation_function_output(ann, FANN_LINEAR_PIECE);
            break;
        case 13:
            fann_set_activation_function_output(ann, FANN_LINEAR_PIECE_SYMMETRIC);
            break;
        case 14:
            fann_set_activation_function_output(ann, FANN_SIN_SYMMETRIC);
            break;
        case 15:
            fann_set_activation_function_output(ann, FANN_COS_SYMMETRIC);
            break;
        case 16:
            fann_set_activation_function_output(ann, FANN_SIN);
            break;
        case 17:
            fann_set_activation_function_output(ann, FANN_COS);
            break;
        default:
            fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
            break;

    }
	//fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	//fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	DIR *pdir = NULL;
	vector<string> v;

	if (volba == 1){
		train_files = "/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/train_data/basic_minutiae";
		pdir = opendir(train_files.c_str());
	}
	else if (volba == 2){
		train_files = "/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/train_data/complex_minutiae";
		pdir = opendir(train_files.c_str());
	}

	struct dirent *pent = NULL;

	if (pdir == NULL) // if pdir wasn't initialised correctly
	{ // print an error message and exit the program
		cout << "\nERROR! pdir could not be initialised correctly";
		exit(3);
	}
	int counter = 0;
	while (pent = readdir(pdir)) // while there is still something in the directory to list
	{
		counter++;
		if (pent == NULL) // if pent has not been initialised correctly
		{ // print an error message, and exit the program
			cout << "\nERROR! pent could not be initialised correctly";
			exit(3);
		}
		if (counter > 2)
			v.push_back(pent->d_name);
	}
	closedir(pdir);

	fann_train_data *data;
	int value;

	data = fann_create_train(v.size(), num_input, num_output);
	for (int k = 0; k < v.size(); k++){
		file = train_files + '/' + v.at(k);
		//cout << file << endl;
		test_dat = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		//imwrite("vzorka.bmp", test_dat);
		int c = 0;

		//postupne nacitanie trenovacich suborov
		for (int i = 0; i < test_dat.rows; i++) {
			for (int j = 0; j < test_dat.cols; j++) {
				value = (int)test_dat.at<uchar>(i, j);
				if (value == 255){
					value = 1;
				}
				if (value == 0) {
					value = -1;
				}
				data->input[k][c] = value;          //zapis hodnot pixelov do vektora
				c++;
			}
		}
		if (volba == 1) {       //priradovanie vystupnych hodnot podla typu markantu
			if (file.find("ending") != string::npos) {
				data->output[k][0] = 1;
				data->output[k][1] = -1;
			}
			else if (file.find("bif") != string::npos) {
				data->output[k][0] = -1;
				data->output[k][1] = 1;
			}
		}
		else if (volba == 2) {
			if (file.find("break") != string::npos) {
				data->output[k][0] = 1;
				data->output[k][1] = -1;
				data->output[k][2] = -1;
			}
			else if (file.find("bridge") != string::npos) {
				data->output[k][0] = -1;
				data->output[k][1] = 1;
				data->output[k][2] = -1;
			}
			else if (file.find("proti") != string::npos) {
				data->output[k][0] = -1;
				data->output[k][1] = -1;
				data->output[k][2] = 1;
			}
		}
	}

	String train_file, network_file;                 //vytvorenie a ulozenie konfiguracnych suborov s NS a datami trenovania
	if (volba == 1) {
	    cout << "Save basic.data & basic.net" << endl;
		train_file = "basic.data";
		network_file = "basic.net";
	}
	else if (volba == 2){
	    cout << "Save complex.data & complex.net" << endl;
		train_file = "complex.data";
		network_file = "complex.net";
	}
	//ulozenie tren. dat do suboru
	fann_save_train(data, train_file.c_str());

	//trenovanie NS
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
	//ulozenie natrenovanej NS do suboru
	fann_save(ann, network_file.c_str());
	fann_destroy(ann);

	cout << "Train NN done     " << endl;
}

//vyhladanie zakladnych markantov pomocou CN
vector<vector<int> > search(Mat img){
    int cn,row_shift,col_shift;
    cn = 0;
    row_shift = img.rows/10;
    col_shift = img.cols/10;

    vector <vector<int> > position;

    for(int k=row_shift; k<img.rows-row_shift; k++) {
        for(int l=col_shift; l<img.cols-col_shift; l++) {
            if(img.at<uchar>(k,l) == 0) {
                cn = abs((img.at<uchar>(k-1,l-1))-(img.at<uchar>(k,l-1)))+abs((img.at<uchar>(k,l-1))-(img.at<uchar>(k+1,l-1)))+abs((img.at<uchar>(k+1,l-1))-(img.at<uchar>(k+1,l)))+abs((img.at<uchar>(k+1,l))-(img.at<uchar>(k+1,l+1)))+abs((img.at<uchar>(k+1,l+1))-(img.at<uchar>(k,l+1)))+abs((img.at<uchar>(k,l+1))-(img.at<uchar>(k-1,l+1)))+abs((img.at<uchar>(k-1,l+1))-(img.at<uchar>(k-1,l)))+abs((img.at<uchar>(k-1,l))-(img.at<uchar>(k-1,l-1)));
                if((cn/255/2 == 1) || (cn/255/2 == 3)) {
                    vector<int> tmp(3);
                    tmp[0] = cn/255/2;
                    tmp[1] = k;
                    tmp[2] = l;
                    position.push_back(tmp);
                 }
            }
        }
    }

    return position;
}

void draw_sign(Mat img) {
    int ending=0;
    int bif=0;
    for(unsigned int i=0; i<positions.size(); i++) {
        Point center(positions[i][2],positions[i][1]);
        if(positions[i][0]==1) {            //ukoncenia
            circle(img,center,3,Scalar(255,0,0),1,8,0);
            ending++;
        } else if(positions[i][0]==3) {     //rozdvojenia
            circle(img,center,6,Scalar(0,255,0),1,8,0);
            bif++;
        }
    }
	//cout << "Pocet ukonceni:   " << ending << endl;
	//cout << "Pocet rozdvojeni: " << bif << endl;
}

Mat crossingNumber(Mat img){
    positions = search(img);
    cvtColor(img, img, CV_GRAY2BGR);
    draw_sign(img);

    return img;
}

void find(string dest){
	input = imread(dest, CV_LOAD_IMAGE_GRAYSCALE);

    output_cn = crossingNumber(input);
	item = output_cn;

	imwrite("complexus.bmp", output_cn);
}

Mat draw_detected(Mat img, vector <vector<int> > detected){
        cvtColor(img, img, CV_GRAY2BGR);
    for(int i=0; i<detected.size(); i++) {
        Point center(detected[i][2],detected[i][1]);
        if(detected[i][0]==1) {            //ukoncenia
            circle(img,center,3,Scalar(255,0,0),1,8,0);
        } else if(detected[i][0]==3) {     //rozdvojenia
            circle(img,center,6,Scalar(0,0,255),1,8,0);
        } else if(detected[i][0]==4) {     //prerusenia
            circle(img,center,15,Scalar(0,0,255),1,8,0);
        } else if(detected[i][0]==5) {     //premostenia
            circle(img,center,15,Scalar(0,255,0),1,8,0);
        } else if(detected[i][0]==6) {     //protilahle rozdvojenia
            circle(img,center,10,Scalar(255,0,0),1,8,0);
        }
    }

    return img;
}

void extract(string netFile, int volba){
	struct fann *ann;
	ann = fann_create_from_file(netFile.c_str());
	int value;
	detected_minutiae.clear();

	int end = 0;
	int bif = 0;
	int prerus = 0;
	int bridge = 0;
	int proti = 0;

	for(int i =0; i<positions.size();i++){
        if(positions[i][0] == 1){
            if(volba == 1){
                double test[9];
                int pix = 0;
                for(int k=(positions[i][1])-1; k<=(positions[i][1])+1; k++) {
                    for(int l=(positions[i][2])-1; l<=(positions[i][2])+1; l++) {
                        if((int)input.at<uchar>(k,l)==255) {
                            test[pix] = 1;
                            } else {
                                test[pix] = -1;
                            }
                            pix++;
                        }
                    }
                    double *output = fann_run(ann, test);
                    //pridanie spravne rozpoznaneho markantu do vystupneho vektora
                    if(output[0]>0.9) {
                        detected_minutiae.push_back(positions[i]);
                        end++;
                    }
                    /*cout << "ukoncenie : " << i << " " << output[0]  << " " << output[1] << endl;*/
                    output = NULL;
            }else if (volba == 2){
                if (((positions[i][1]-25) >= 0)&&
                        ((positions[i][2]-25) >= 0)&&
                        ((positions[i][1]+25) < input.rows)&&
                        ((positions[i][2]+25) < input.cols)) {
                        double test[2601];
                        int pix = 0;                                    //nacitanie bloku obrazu okolo suradnice
                        for(int k=(positions[i][1])-25; k<=(positions[i][1])+25; k++) {
                            for(int l=(positions[i][2])-25; l<=(positions[i][2])+25; l++) {
                                if((int)input.at<uchar>(k,l)==255) {
                                    test[pix] = 1;
                                } else {
                                    test[pix] = -1;
                                }
                                pix++;
                            }
                        }
                        double *output = fann_run(ann, test);
                        //pridanie spravne rozpoznaneho markantu do vystupneho vektora
                        if((output[0]>0.98) && (output[1]<-0.8) && (output[2]<-0.8)) {
                            prerus++;
                            vector<int> tmp(3);
                            tmp[0] = 4;         //oznacenie prerusenia pre vykreslovanie
                            tmp[1] = positions[i][1];
                            tmp[2] = positions[i][2];
                            detected_minutiae.push_back(tmp);
                        }
                        /*cout << "prerusenie : " << i << " " << output[0]  << " " << output[1] << " " << output[2] << endl;*/
                        }
                    }
                } else if (positions[i][0] == 3){
                    if (volba == 1){
                        double test[9];
                        int pix = 0;
                        for(int k=(positions[i][1])-1; k<=(positions[i][1])+1; k++) {
                            for(int l=(positions[i][2])-1; l<=(positions[i][2])+1; l++) {
                                if((int)input.at<uchar>(k,l)==255) {
                                    test[pix] = 1;
                                } else {
                                    test[pix] = -1;
                                }
                                pix++;
                            }
                        }
                        double *output = fann_run(ann, test);
                        //pridanie spravne rozpoznaneho markantu do vystupneho vektora
                        if(output[1]>0.9) {
                            detected_minutiae.push_back(positions[i]);
                            bif++;
                        }
                        /*cout << "rozdvojenie : " << i << " " << output[0] << " " << output[1] << endl;*/
                    } else if (volba == 2){                     //extrakcia protilahlych rozdvojeni a premosteni
                        if (((positions[i][1]-25) >= 0)&&
                            ((positions[i][2]-25) >= 0)&&
                            ((positions[i][1]+25) < input.rows)&&
                            ((positions[i][2]+25) < input.cols)) {
                            double test[2601];
                            int pix = 0;
                            for(int k=(positions[i][1])-25; k<=(positions[i][1])+25; k++) {
                                for(int l=(positions[i][2])-25; l<=(positions[i][2])+25; l++) {
                                    if((int)input.at<uchar>(k,l)==255) {
                                        test[pix] = 1;
                                    } else {
                                        test[pix] = -1;
                                    }
                                    pix++;
                                }
                            }
                            double *output = fann_run(ann, test);
                            //pridanie spravne rozpoznaneho markantu do vystupneho vektora
                            if((output[0]<-0.8) && (output[1]>0.98) && (output[2]<-0.8)) {
                                bridge++;
                                vector<int> tmp(3);
                                tmp[0] = 5;         //oznacenie premostenia pre vykreslovanie
                                tmp[1] = positions[i][1];
                                tmp[2] = positions[i][2];
                                detected_minutiae.push_back(tmp);
                                /*cout << "premostenie : " << i << " " << output[0]  << " " << output[1]
                                                                     << " " << output[2] << endl;*/
                            } else if((output[0]<-0.8) && (output[1]<-0.8) && (output[2]>0.98)) {
                                proti++;
                                vector<int> tmp(3);
                                tmp[0] = 6;         //oznacenie protilahleho rozdvojenia pre vykreslovanie
                                tmp[1] = positions[i][1];
                                tmp[2] = positions[i][2];
                                detected_minutiae.push_back(tmp);
                                /*cout << "protilahle rozdvojenia : " << i << " " << output[0]  << " " << output[1]
                                                                                << " " << output[2] << endl;*/
                            }

                        }
                    }
                }
            }
            output_cn = draw_detected(input,detected_minutiae);           //vykreslenie detekovanych markantov
            if(volba == 1){
                imwrite("basic.bmp",output_cn);
            } else if (volba == 2){
                imwrite("complex.bmp", output_cn);
            }
            item = output_cn;
        }

void train_basic(){
	const unsigned int num_input = 9;
	const unsigned int num_output = 2;
	const unsigned int num_layers = 5;
	const unsigned int num_neurons_hidden1 = 3;
	const unsigned int num_neurons_hidden2 = 1;
	const unsigned int num_neurons_hidden3 = 1;
	const float desired_error = (const float) 0.0001;
	const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 100;

	struct fann *ann = fann_create_standard(num_layers, num_input,
		num_neurons_hidden1, num_neurons_hidden2, num_neurons_hidden3, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
	fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
	//fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, "train-end-bifurcation.data", max_epochs,
		epochs_between_reports, desired_error);

	fann_save(ann, "net-basic.net");

	fann_destroy(ann);

	system("PAUSE");
}

void train_complex(){
	const unsigned int num_input = 51*51;
	const unsigned int num_output = 3;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden1 = 3500;
	const unsigned int num_neurons_hidden2 = 480;
	const unsigned int num_neurons_hidden3 = 240;
	const float desired_error = (const float) 0.0001;
	const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 100;

	struct fann *ann = fann_create_standard(num_layers, num_input,
		num_neurons_hidden1, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
	fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
	//fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, "complex.data", max_epochs,
		epochs_between_reports, desired_error);

	fann_save(ann, "complex.net");

	fann_destroy(ann);

	//system("PAUSE");
}

Scalar getMarkantColor(string markant) {
	if (markant == "RIDGE_ENDING") return Scalar(0, 0, 0);
	if (markant == "BIFURCATION") return Scalar(0, 0, 128);
	if (markant == "FRAGMENT") return Scalar(0, 128, 0);
	if (markant == "POINT_OR_DOT") return Scalar(128, 0, 0);
	if (markant == "BREAK") return Scalar(0, 0, 256);
	if (markant == "ENCLOSURE") return Scalar(0, 256, 0);
	if (markant == "CROSSBAR") return Scalar(256, 0, 0);
	if (markant == "BRIDGE") return Scalar(0, 128, 128);
	if (markant == "OPPOSITED_BIFURCATION") return Scalar(128, 0, 128);
	if (markant == "OVERLAP") return Scalar(256, 0, 128);
	if (markant == "Y_OR_M") return Scalar(128, 128, 0);
	if (markant == "DOCK") return Scalar(256, 128, 0);
	if (markant == "RETURN") return Scalar(0, 128, 256);
	return Scalar(256, 256, 256);
}

void CrossinNumber(Mat image){
	int count = 0;
	ending = Mat::zeros(image.rows, image.cols, image.type());
	bifuraction = Mat::zeros(image.rows, image.cols, image.type());
	int cn = 0;
	for (int i = 1; i < image.rows - 1; i++){
		for (int j = 1; j < image.cols - 1; j++){
			if (image.at<uchar>(i, j) == 0){
				cn = (abs(image.at<uchar>(i - 1, j - 1) - image.at<uchar>(i, j - 1))
					+ abs(image.at<uchar>(i, j - 1) - image.at<uchar>(i + 1, j - 1))
					+ abs(image.at<uchar>(i + 1, j - 1) - image.at<uchar>(i + 1, j))
					+ abs(image.at<uchar>(i + 1, j) - image.at<uchar>(i + 1, j + 1))
					+ abs(image.at<uchar>(i + 1, j + 1) - image.at<uchar>(i, j + 1))
					+ abs(image.at<uchar>(i, j + 1) - image.at<uchar>(i - 1, j + 1))
					+ abs(image.at<uchar>(i - 1, j + 1) - image.at<uchar>(i - 1, j))
					+ abs(image.at<uchar>(i - 1, j) - image.at<uchar>(i - 1, j - 1))) / 2 / 255;
				if (cn == 3) bifuraction.at<uchar>(i, j) = 1;
				if (cn == 1) ending.at<uchar>(i, j) = 1;
			}
			count = 0;
		}
	}
}

Mat runNeuralBasic(Mat origImage, Mat tmpImage, string netFile){
	fstream f;
	f.open(netFile.c_str(), fstream::in);

	fann_type *out;
	struct fann *ann = fann_create_from_file(netFile.c_str());
	unsigned int *lays = (unsigned int *)malloc(fann_get_num_layers(ann) * sizeof(unsigned int));
	fann_get_layer_array(ann, lays);
	int block = sqrt(lays[0]);
	long long help = 0;

	fann_type *input = (fann_type *)malloc(block * block * sizeof(fann_type));
	string markant = "RIDGE_ENDING";
	CrossinNumber(origImage);

	for (int i = 0; i < ending.rows; i++){
		for (int j = 0; j < ending.cols; j++) {
			if ((int)ending.at<uchar>(i, j) == 1 || (int)bifuraction.at<uchar>(i, j) == 1) {
				if ((int)ending.at<uchar>(i, j) == 1) markant = "RIDGE_ENDING";
				if ((int)bifuraction.at<uchar>(i, j) == 1) markant = "BIFURCATION";
				help = 0;
				for (int k = i - 1; k <= i + 1; k++){
					for (int l = j - 1; l <= j + 1; l++) {
						if (origImage.at<uchar>(k, l) == 255) input[help] = 1;
						else input[help] = -1;
						help++;
					}
				}
				out = fann_run(ann, input);
				string hlp = "";
				if (out[0] > 0.7) hlp.append("1 ");
				else hlp.append("0 ");
				if (out[1] > 0.7) hlp.append("1");
				else hlp.append("0");
				if (hlp != "0 0" && hlp != "1 1") {
					Point point(j, i);
					circle(tmpImage, point, 5, getMarkantColor(markant), 1, 20);
				}
			}

		}
	}

	fann_destroy(ann);
	return tmpImage;
}

Mat runNeuralComplex(Mat origImage, Mat tmpImage, string netFile) {
	fstream f;
	f.open(netFile.c_str(), fstream::in);
	fann_type *out;
	//double *out;
	//const int size = hidLayers.size();

	struct fann *ann = fann_create_from_file(netFile.c_str());
	unsigned int *lays = (unsigned int *)malloc(fann_get_num_layers(ann) * sizeof(unsigned int));
	fann_get_layer_array(ann, lays);
	int block = sqrt(lays[0]);
	long long help = 0;
	fann_type *input = (fann_type *)malloc(block * block * sizeof(fann_type));
	//double *input = (double *)malloc(block * block * sizeof(double));
	string markant = "RIDGE_ENDING";
	CrossinNumber(origImage);
	for (int i = block / 2; i < ending.rows - (block / 2); i++){
		for (int j = block / 2; j < ending.cols - (block / 2); j++) {
			if ((int)bifuraction.at<uchar>(i, j) == 1 || (int)ending.at<uchar>(i, j) == 1) {
				if ((int)bifuraction.at<uchar>(i, j) == 1) markant = "BIFURCATION";
				help = 0;
				for (int k = i - (block / 2); k <= i + (block / 2); k++){
					for (int l = j - (block / 2); l <= j + (block / 2); l++) {
						/*for (int k = 0; k < 7; k++){
						for (int l = 0; l < 7; l++) {*/
						if (origImage.at<uchar>(k, l) == 255) input[help] = 1;
						else input[help] = -1;
						help++;
					}
				}
				//display(input, block * block);
				out = fann_run(ann, input);
				string hlp = "";
				for (int i = 0; i < 3; i++) {
					if (out[i] > 0.7) hlp.append("1 ");
					else hlp.append("0 ");
				}
				if (out[3] > 0.7) hlp.append("1");
				else hlp.append("0");
				/* if (hlp == "1 0 " || hlp == "0 1 ") {
				if (hlp == "1 0 ") markant = "BREAK";
				else markant = "CROSSBAR";*/
				//if (hlp != "0 0 0 0" && hlp != "1 1 1 1") {
				if (hlp == "0 0 0 1" || hlp == "1 0 0 0") {
					if (hlp == "0 0 0 1")
						markant = "BREAK";
					else
						markant = "CROSSBAR";
					cv::Point point(j, i);
					//cv::Point point(j, i);
					circle(tmpImage, point, 10, getMarkantColor(markant), 1, 0);
					//cv::Point point1(j + 5, i + 5);
					//rectangle(tmpImage, point, point1, getMarkantColor(markant));
				}
			}
		}
	}
	fann_destroy(ann);
	return tmpImage;
}



void FalseMarkant(Mat image, Mat mask){
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			if (ending.at<uchar>(i, j) == 1){

				for (int k = i - 15; k <= i + 15; k++){
					for (int l = j - 15; l < j + 15; l++){
						if (mask.at<uchar>(k, l) == 0){
							ending.at<uchar>(i, j) = 0;
						}
					}
				}
			}

			if (bifuraction.at<uchar>(i, j) == 1){

				for (int k = i - 15; k <= i + 15; k++){
					for (int l = j - 15; l < j + 15; l++){
						if (mask.at<uchar>(k, l) == 0)
							bifuraction.at<uchar>(i, j) = 0;
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{

	//train_basic();

	//train_complex();

	//nacitanie vstupnych parametrov

	int input_basic = atoi(argv[4]);                //argv[4]
    int output_basic = atoi(argv[5]);               //argv[5]
    int layers_basic = atoi(argv[6]);               //argv[6]

    int input_complex = atoi(argv[7]);             //argv[7]
    int output_complex = atoi(argv[8]);             //argv[8]
    int layers_complex = atoi(argv[9]);             //argv[9]

    float desired_error = atof(argv[10]);            //argv[10]
    int max_epochs = atoi(argv[11]);                 //argv[11]
    int hidden_neuron1 = atoi(argv[12]);             //argv[12]
    int hidden_neuron2 = atoi(argv[13]);             //argv[13]
    int activation_func_hidden = atoi(argv[14]);     //argv[14]
    int activation_func_output = atoi(argv[15]);     //argv[15]

    int size_of_block_oriantation = atoi(argv[16]);  //argv[16]
    int size_of_block_gabor = atoi(argv[17]);        //argv[17]
    int sigma = atoi(argv[18]);                      //argv[18]
    double lambda = (double)atof(argv[19]);                     //argv[19]
    double gamma = (double)atof(argv[20]);                      //argv[20]

    clock_t begin, end;
	double elapsed_secs;
	long minutes, seconds,total_min,total_sec;
	long sum_min = 0;
	long sum_sec = 0;

    /*cout << "Parameters for processing NN" << endl;
    cout << "Number of inputs for basic NN: " << input_basic << endl;
    cout << "Number of outputs for basic NN: " << output_basic << endl;
    cout << "Number of layers for basic: " << layers_basic << endl;

    cout << "Number of inputs for complex NN: " << input_complex << endl;
    cout << "Number of outputs for complex NN: " << output_complex << endl;
    cout << "Number of layers for complex NN: " << layers_complex << endl;

    cout << "Desired error for NN: " << desired_error << endl;
    cout << "Max number of epochs: " << max_epochs << endl;
    cout << "Number of hidden neurons for basic NN: " << hidden_neuron1 << endl;
    cout << "Number of hidden neurons for complex NN: " << hidden_neuron2 << endl;
    cout << "Activation function for hidden: " << activation_func_hidden << endl;
    cout << "Activation function for output: " << activation_func_output << endl;

    cout << "\nParameters for preprocessing" << endl;
    cout << "Size of block for orientation map: " <<size_of_block_oriantation << endl;
    cout << "Size of block for Gabor filter: " << size_of_block_gabor << endl;
    cout << "Parameter sigma for Gabor filter: " << sigma << endl;
    cout << "Parameter lambda for Gabor filter: " << lambda << endl;
    cout << "Parameter gamma for Gabor filter: " <<gamma << endl;
    cout << endl;*/
    //cin.get();

	        //learn_NN(1, 9, 2, 3, 15, 0.0001, 5000, 5, 5);

	Mat outImageBasic, outImageComplex, outImageOrig;

    stringstream sstm;


    //nastavenie kde sa maju ukladat vystupy a kde su ulozene obrazky
    int id = 0;
    string dest = "/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/";
    string destin = "/home/gregor/projects/dp_image_processing/dp_iamge_processing/media/tasks/";
    destin +=string(argv[1]);
    string result;
    int dir_err;

    /*do
    {
        id++;
        sstm.str("");
        sstm << desti << id;
        result = sstm.str();
        dir_err = mkdir(result.c_str(), 0777);

    }while (-1 == dir_err);

    //result += '/';
    dest = result.c_str();*/
    result = dest.c_str();
    result += string(argv[3]);

    cout << "_________________Fingerprint Processing__________________" << endl;
    cout << "Task ID: " << string(argv[1]) <<endl;

	Mat image = imread(result);
	string imageFile = result;

	bool invalid = false;

    if (image.empty())
	{
		cout << "Cannot load image!" << endl;
		return -1;
	}

	int rows = image.rows;
	int cols = image.cols;

	Size s = image.size();
	rows = s.height;
	cols = s.width;

	cout << "Picture resolution: " << cols << "x" << rows << endl;
	//system("PAUSE");

    //kontrola vstupneho obrazka podla hodnoty pixelov - pre NN musi byt 0 a 255
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if ((int)image.at<uchar>(i, j) != 255 && (int)image.at<uchar>(i, j) != 0) {
				printf("Invalid file input -> Image must containst only 255 or 0!\n");
				invalid = true;
				break;
			}
		}
		if (invalid == true)
			break;
	}


    if(string(argv[2]) == "0"){

        //trenovanie neuronovej siete najprv na zakladne markanty a potom na komplexne
        begin = clock();
        learn_NN(1, input_basic, output_basic, layers_basic, hidden_neuron1, desired_error, max_epochs, activation_func_hidden, activation_func_output);
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        seconds = (int)elapsed_secs % 60;
        minutes = (int)elapsed_secs / 60;
        sum_min += minutes;
        sum_sec += seconds;
        cout << "Training time: "<< minutes << ":" << seconds << endl;

        begin = clock();
        learn_NN(2, input_complex, output_complex, layers_complex, hidden_neuron2, desired_error, max_epochs, activation_func_hidden, activation_func_output);
        end = clock();
        elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        seconds = (int)elapsed_secs % 60;
        minutes = (int)elapsed_secs / 60;
        sum_min += minutes;
        sum_sec += seconds;
        cout << "Training time: " <<minutes << ":" << seconds << endl;
	}

	Preprocessing preprocessing(image, cols, rows);

	//Segmentacia
	begin = clock();
	cout << "\nStart preprocessing" << endl;
	cout << "Segmentation...";
	preprocessing.Segmentation(preprocessing.Get_gray_image());
	cout << "Done	";
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	seconds = (int)elapsed_secs % 60;
	minutes = (int)elapsed_secs / 60;
	sum_min += minutes;
    sum_sec += seconds;
	cout << minutes << ":" << seconds << endl;

	if (invalid == true){
		//Orientacna mapa
		begin = clock();
		cout << "Orientation Map...";
		preprocessing.Set_sizeofBlockOrientation(size_of_block_oriantation);
		preprocessing.OrientationMap(preprocessing.Get_segmentation());
		cout << "Done	";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;

		//Gaborov filter
		begin = clock();
		cout << "Gabor Filter...";
		preprocessing.Set_sigma(sigma);
		preprocessing.Set_lambda(lambda);
		preprocessing.Set_gamma(gamma);
		preprocessing.Set_sizeofBlockGabor(size_of_block_gabor);
		preprocessing.GaborFilter(preprocessing.Get_segmentation());
		//preprocessing.ColourGabor(preprocessing.Get_gabor_filter());
		cout << "Done	";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;

		//Binarizacia
		begin = clock();
		cout << "Binarization...";
		preprocessing.Binarization(preprocessing.Get_gabor_filter());
		cout << "Done	";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;

		//Zuzenie papilarnych linii
		begin = clock();
		cout << "Thinning...";
		preprocessing.ThinningImage(preprocessing.Get_binarization());
		cout << "Done	";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;

		//Crossin Number
		begin = clock();
		cout << "Crossin number...";
		CrossinNumber(preprocessing.Get_thinning_image());
		cout << "Done	";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;

        //ulozenie vystupov
        result = destin.c_str();
        result += "/output/1segmentation.jpg";
		imwrite(result.c_str(), preprocessing.Get_segmentation());

		result =destin.c_str();
		result += "/output/2raw_mask.jpg";
		imwrite(result.c_str(), preprocessing.Get_raw_mask());

		result =destin.c_str();
		result += "/output/3mask.jpg";
		imwrite(result.c_str(), preprocessing.Get_mask());

		result =destin.c_str();
		result += "/output/4orientation_map.jpg";
		imwrite(result.c_str(), preprocessing.DrawOrientationMap(preprocessing.Get_orientation()));

		result =destin.c_str();
		result += "/output/5gabor.jpg";
		imwrite(result.c_str(), preprocessing.Get_gabor_filter());

		result =destin.c_str();
		result += "/output/6binarization.jpg";
		imwrite(result.c_str(), preprocessing.Get_binarization());

		result =destin.c_str();
		result += "/output/7thinning.jpg";
		imwrite(result.c_str(), preprocessing.Get_thinning_image());

		cvtColor(image, image, CV_RGB2GRAY);
		outImageBasic = preprocessing.Get_thinning_image();
		outImageComplex = preprocessing.Get_thinning_image();;
		outImageOrig = image;
		image = preprocessing.Get_thinning_image();

		cvtColor(outImageBasic, outImageBasic, CV_GRAY2BGR);
		cvtColor(outImageOrig, outImageOrig, CV_GRAY2BGR);
		cvtColor(outImageComplex, outImageComplex, CV_GRAY2BGR);

        begin = clock();
		cout << "Run neural Basic...";
		outImageBasic = runNeuralBasic(image, outImageBasic, "basic.net");
		outImageOrig = runNeuralBasic(image, outImageOrig, "basic.net");
		cout << "Done    ";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;
		//cout << "Press enter to continue...";
        //cin.get();
        begin = clock();
		cout << "Run neural Complex...";
		outImageComplex = runNeuralComplex(image, outImageComplex, "complex.net");
		cout << "Done    ";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;
//		getchar();

        result =destin.c_str();
		result += "/output/8output_basic_nn.jpg";
		imwrite(result.c_str(), outImageBasic);
		//imshow("Image Out", outImageBasic);

        result =destin.c_str();
		result += "/output/9output_basic_nn_original.jpg";
		imwrite(result.c_str(), outImageOrig);
		//imshow("Image Out2", outImageOrig);

		result =destin.c_str();
		result += "/output/91output_complex_nn.jpg";
		imwrite(result, outImageComplex);
	}
	else{
		cvtColor(image, image, CV_RGB2GRAY);
		outImageBasic = image;
		cvtColor(outImageBasic, outImageBasic, CV_GRAY2BGR);
		outImageComplex = image;
		cvtColor(outImageComplex, outImageComplex, CV_GRAY2BGR);

        begin = clock();
		cout << "Run neural Basic...";
		outImageBasic = runNeuralBasic(image, outImageBasic, "basic.net");
		cout << "Done    ";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;
		//cout << "Press enter to continue...";
        //cin.get();

        begin = clock();
		cout << "Run neural Complex...";
		outImageComplex = runNeuralComplex(image, outImageComplex, "complex.net");
		cout << "Done    ";
		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		seconds = (int)elapsed_secs % 60;
		minutes = (int)elapsed_secs / 60;
		sum_min += minutes;
        sum_sec += seconds;
		cout << minutes << ":" << seconds << endl;
		//cout << "Press enter to continue...";
        //cin.get();

		FalseMarkant(outImageBasic, preprocessing.Get_mask());
		//imshow("Imge original", image);

        result =destin.c_str();
		result += "/output/1output_basic_nn.jpg";
		imwrite(result.c_str(), outImageBasic);
		//imshow("Image Out", outImageBasic);

        result =destin.c_str();
		result += "/output/2output_complex_nn.jpg";
		imwrite(result, outImageComplex);
		//imshow("Image Out Complex", outImageComplex);

        /*find(imageFile);
		cout << "Running extract..." << endl;
		extract("basic.net", 1);
		cout << "basic done" << endl;

		extract("complex.net", 2);
		cout << "complex done" << endl;*/

	}

	total_min = (int)sum_min + ((int)sum_sec / 60);
	total_sec = (int)sum_sec % 60;

	cout << "\nTotal running time: " << total_min << ":" << total_sec << endl;

	waitKey(0);

	return 1;
}
