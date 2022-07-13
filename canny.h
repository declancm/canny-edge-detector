#pragma once

#include <math.h>

#include <iostream>
// #include <fstream>

#include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

void cannyEdgeDetection(std::string readLocation, std::string writeLocation, double lowerThreshold, double higherThreshold);
int* imgToArray(cv::Mat img, int sizeRows, int sizeCols, int sizeDepth, uint8_t* pixelPtr);
void arrayToImg(int* pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
int* gaussianBlur(int* pixels, double kernel[5][5], double kernelConst, int sizeRows, int sizeCols, int sizeDepth);
int* rgbToGrayscale(int* pixels, int sizeRows, int sizeCols, int sizeDepth);
int* cannyFilter(int* pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth, double lowerThreshold, double higherThreshold);
