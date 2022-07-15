#pragma once

#include <math.h>

#include <iostream>
#include <vector>
// #include <fstream>

#include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

void cannyEdgeDetection(std::string readLocation, std::string writeLocation, double lowerThreshold, double higherThreshold);
std::vector<int> imgToArray(cv::Mat img, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
void arrayToImg(std::vector<int>& pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
std::vector<int> gaussianBlur(std::vector<int>& pixels, std::vector<std::vector<double>>& kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth);
std::vector<int> rgbToGrayscale(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth);
std::vector<int> cannyFilter(std::vector<int>& pixels, int sizeRows, int sizeCols, int sizeDepth, double lowerThreshold, double higherThreshold);
