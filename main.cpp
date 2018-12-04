#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 90;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

bool FindRingPattern(vector<Point2f> &probableCPs, int num_rows, int num_cols) {
	int n = probableCPs.size();
	std::vector<Vec4f> lines;
	// Generamos todas las Combinaciones de "Lineas" en grupos de 5 posibles
	// Es 5 xq seleccionamos las lineas
	std::vector< std::vector<int> > combinations = GenerateCombinations(probableCPs.size(), num_cols);

	//Aprovechamos las lineas temporales y Selecionamos las que tengan Baja Desviacion
	std::vector<Vec4f> preSelectedLines;
	std::vector<std::vector<int> > combination_preSelectedLines;
	for (int i = 0; i < combinations.size(); i++) {
		std::vector<Point2f> tmpPoints(num_cols);
		Vec4f tmpLine;
		for (int j = 0; j < num_cols; j++) {
			tmpPoints[j] = probableCPs[combinations[i][j]];
		}
		fitLine(tmpPoints, tmpLine, CV_DIST_L2, 0, 0.01, 0.01);
		// Extraction of Features
		//Le damos forma a los valores vectoriales que nos devuelve fitline
		// r = a + r*b --> p0 punto de paso, v vector director normalizado
		float vx = tmpLine[0], vy = tmpLine[1], x0 = tmpLine[2], y0 = tmpLine[3];
		Point2f a = Point2f(x0, y0), b = Point2f(vx, vy);

		float m = 80.0;

		std::vector<float> distances;
		for (int k = 0; k < num_cols; k++) {
			//Calculamos la distancia del punto a la recta y almacenamos para el calculo de la desviacion
			float t = (tmpPoints[k].dot(b) - a.dot(b)) / (cv::norm(b) * cv::norm(b));
			float dist = cv::norm(tmpPoints[k] - (a + t * b));
			distances.push_back(dist);
		}

		float stddev = StandarDesviation(distances);

		//Si el error de la linea no es mucho. Seleccionamos la linea
		if (stddev < 0.5f) {
			preSelectedLines.push_back(tmpLine);
			//Guardamos la Combinacion
			combination_preSelectedLines.push_back(combinations[i]);
		}

	}

	// Apply some filters here to verify line selection
	// Then Order Points and Store in CPs(Hard verification of only 20 Ordered and Aligned Control Points)
	// Acordemonos que ya seleccionamos solo lineas con 5 puntos
	if (preSelectedLines.size() == 4) {
		//Tenemos que ordenar las lineas. (Recordemos que son lineas paralelas)
		//Primero verificamos la pendiente

		//LINE ORDERING
			//Recordemos la grilla que presenta openCV 
			// -------> x+
			// |
			// |
			// y+

		Vec4f Line = preSelectedLines[0];
		float vx = Line[0], vy = Line[1], x0 = Line[2], y0 = Line[3];
		//Pendiente
		float slope = vy / vx;
		if (abs(slope) < 5.0f) { //Evaluamos las pendientes de casi 80 grados (Revisar esta funcion)
			std::vector<float> y_intersection(4);
			//Calcular el punto de interseccion por el eje y
			for (int i = 0; i < 4; i++) {
				Vec4f tmpLine = preSelectedLines[0];
				float vx = tmpLine[0], vy = tmpLine[1], x0 = tmpLine[2], y0 = tmpLine[3];

				float t = -x0 / vx;
				float y = y0 + t * vy;

				y_intersection[i] = y;
			}

			//Realizamos un bubble sort en base a las intersecciones con el eje y
			//ordenamiento por burbuja
			bool swapp = true;
			while (swapp)
			{
				swapp = false;
				for (int i = 0; i < preSelectedLines.size() - 1; i++)
				{
					if (y_intersection[i] > y_intersection[i + 1]) {
						//Cambiamos en todos nuestros vectores
						std::swap(y_intersection[i], y_intersection[i + 1]);
						std::swap(preSelectedLines[i], preSelectedLines[i + 1]);
						std::swap(combination_preSelectedLines[i], combination_preSelectedLines[i + 1]);
						swapp = true;
					}
				}
			}// Fin del ordenamiento

			// Para Cada Linea obtener los CP segun la combinacion y ordenarlos por el eje X
			// Obtenemos los puntos desde el CP

			std::vector<Point2f> tmpCPs;
			for (int i = 0; i < num_rows; i++) {
				std::vector<Point2f> tmpCenters(num_cols);
				for (int j = 0; j < num_cols; j++) {
					tmpCenters[j] = probableCPs[combination_preSelectedLines[i][j]];
				}
				sort(tmpCenters.begin(), tmpCenters.end(), cmpx);
				for (int j = 0; j < num_cols; j++) {
					tmpCPs.push_back(tmpCenters[j]);
				}
			}

			probableCPs.clear();
			probableCPs = tmpCPs;

			return true;
		}
	}
	return false;
}// fin de funcion importante

struct node {
	Point2f center;
	int count;
	float w;

	node(Point center, int count, int w)
	{
		this->center = center;
		this->count = count;
		this->w = w;
	}
};

/** @function main */
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	src = imread("images/Captura16.PNG", 1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	// 0: Binary
	// 1: Binary Inverted
	// 2: Threshold Truncated
	// 3: Threshold to Zero
	// 4: Threshold to Zero Inverted

	threshold(src_gray, src_gray, 100, 255, 3);

	/// Create Window
	//char* source_window = "Source";
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);

	createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, 100, 255, THRESH_BINARY);

	/// Detect edges using canny
	Canny(threshold_output, threshold_output, thresh, thresh * 2, 3);

	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the rotated rectangles and ellipses for each contour
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}

	/// Draw contours + rotated rects + ellipses
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Point2f last(-10, -10);
	std::vector<node> centers;
	int n = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// contour
		//drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		// ellipse
		Point2f np(minEllipse[i].center.x, minEllipse[i].center.y);
		if (norm(last - np) < 5) {
			centers[n - 1].count++;
			float w_np = pow(minEllipse[i].size.area(), 3);
			centers[n - 1].center = (np*w_np + centers[n - 1].center*centers[n - 1].w) / (w_np + centers[n - 1].w);
			if (centers[n - 1].count > 2) {
				ellipse(drawing, minEllipse[i], color, 2, 8);
			}
		}
		else {
			n++;
			node nd(np, 1, pow(minEllipse[i].size.area(), 3));
			centers.push_back(nd);
		}
		last = np;
		// rotated rectangle
		/*Point2f rect_points[4]; minRect[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);*/
	}

	std::vector<Point2f> PointBuffer;

	for (auto center : centers) {
		if (center.count > 3)
			PointBuffer.push_back(center.center);
	}

	cv::drawChessboardCorners(src, Size(6, 5), PointBuffer, true);
	namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
	imshow("Pattern", src);

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}