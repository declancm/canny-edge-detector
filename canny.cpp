#include "canny.h"

void cannyEdgeDetection(std::string readLocation, std::string writeLocation, double lowerThreshold, double higherThreshold) {
    if (readLocation == writeLocation) {
        std::cout << "The read file and save file locations cannot be the same.\n";
        return;
    }
    cv::Mat img = cv::imread(readLocation);

    // READ_FILE:
    uint8_t* pixelPtr = (uint8_t*)img.data;
    int sizeRows = img.rows;
    int sizeCols = img.cols;
    int sizeDepth = img.channels();
    int* pixels = imgToArray(img, sizeRows, sizeCols, sizeDepth, pixelPtr);

    arrayToImg(pixels, pixelPtr, sizeRows, sizeCols, sizeDepth);
    // cv::imshow("Original", img);

    // GAUSSIAN_BLUR:

    double kernel[5][5] = {{2.0, 4.0, 5.0, 4.0, 2.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {5.0, 12.0, 15.0, 12.0, 5.0},
                           {4.0, 9.0, 12.0, 9.0, 4.0},
                           {2.0, 4.0, 5.0, 4.0, 2.0}};
    double kernelConst = (1.0 / 159.0);
    int* pixelsBlur = gaussianBlur(pixels, kernel, kernelConst, sizeRows, sizeCols, sizeDepth);

    arrayToImg(pixelsBlur, pixelPtr, sizeRows, sizeCols, sizeDepth);
    // cv::imshow("Blurred", img);

    // GRAYSCALE:

    cv::Mat imgGrayscale(sizeRows, sizeCols, CV_8UC1, cv::Scalar(0));
    uint8_t* pixelPtrGray = (uint8_t*)imgGrayscale.data;

    int* pixelsGray = rgbToGrayscale(pixels, sizeRows, sizeCols, sizeDepth);
    arrayToImg(pixelsGray, pixelPtrGray, sizeRows, sizeCols, 1);
    // cv::imshow("Grayscale", imgGrayscale);

    // CANNY_FILTER:
    int* pixelsCanny = cannyFilter(pixelsGray, pixelPtrGray, sizeRows, sizeCols, 1, lowerThreshold, higherThreshold);
    arrayToImg(pixelsCanny, pixelPtrGray, sizeRows, sizeCols, 1);

    cv::imshow("CannyEdgeDetection", imgGrayscale);
    cv::waitKey(0);

    cv::imwrite(writeLocation, imgGrayscale);

    delete[] pixels;
    delete[] pixelsGray;
    delete[] pixelsCanny;
}

int* imgToArray(cv::Mat img, int sizeRows, int sizeCols, int sizeDepth, uint8_t* pixelPtr) {
    int* pixels = new int[sizeRows * sizeCols * sizeDepth];
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    // converting BGR to RGB
                    pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + 2 - k];
            }
        }
    }
    return pixels;
}

void arrayToImg(int* pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth) {
    int* result = new int[sizeRows * sizeCols * sizeDepth];
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                pixelPtr[i * sizeCols * sizeDepth + j * sizeDepth + k] =
                    pixels[i * sizeCols * sizeDepth + j * sizeDepth + (sizeDepth - 1 - k)];
            }
        }
    }
    return;
}

int* gaussianBlur(int* pixels, double kernel[5][5], double kernelConst, int sizeRows, int sizeCols, int sizeDepth) {
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            for (int k = 0; k < sizeDepth; k++) {
                double sum = 0;
                double sumKernel = 0;
                for (int y = -2; y <= 2; y++) {
                    for (int x = -2; x <= 2; x++) {
                        if ((i + x) >= 0 && (i + x) < sizeRows && (j + y) >= 0 && (j + y) < sizeCols) {
                            double channel = (double)pixels[(i + x) * sizeCols * sizeDepth + (j + y) * sizeDepth + k];
                            sum += channel * kernelConst * kernel[x + 2][y + 2];
                            sumKernel += kernelConst * kernel[x + 2][y + 2];
                        }
                    }
                }
                pixels[i * sizeCols * sizeDepth + j * sizeDepth + k] = (int)(sum / sumKernel);
            }
        }
    }
    return pixels;
}

int* rgbToGrayscale(int* pixels, int sizeRows, int sizeCols, int sizeDepth) {
    int* pixelsGray = new int[sizeRows * sizeCols];
    for (int i = 0; i < sizeRows; i++) {
        for (int j = 0; j < sizeCols; j++) {
            int sum = 0;
            for (int k = 0; k < sizeDepth; k++) {
                sum = sum + pixels[i * sizeCols * sizeDepth + j * sizeDepth + k];
            }
            pixelsGray[i * sizeCols + j] = (sum / sizeDepth);
        }
    }
    return pixelsGray;
}

int* cannyFilter(int* pixels, uint8_t* pixelPtrGray, int sizeRows, int sizeCols, int sizeDepth, double lowerThreshold, double higherThreshold) {
    // intializing before canny
    int* pixelsCanny = new int[sizeRows * sizeCols];
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double* G = new double[sizeRows * sizeCols];
    int* direction = new int[sizeRows * sizeCols];
    int* theta = new int[sizeRows * sizeCols];
    double largestG = 0;

    // perform canny edge detection on everything but the edges
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            // find gx and gy for each pixel
            double gxValue = 0;
            double gyValue = 0;
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    gxValue = gxValue + (gx[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                    gyValue = gyValue + (gy[1 - x][1 - y] * (double)(pixels[(i + x) * sizeCols + j + y]));
                }
            }

            // calculate G and theta
            G[i * sizeCols + j] = std::sqrt(std::pow(gxValue, 2) + std::pow(gyValue, 2));
            double atanResult = atan2(gyValue, gxValue) * 180.0 / 3.14159265;
            theta[i * sizeCols + j] = (int)(180.0 + atanResult);

            if (G[i * sizeCols + j] > largestG) { largestG = G[i * sizeCols + j]; }

            // setting the edges
            if (i == 1) {
                G[i * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            if (j == 1) {
                G[(i - 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j] = theta[i * sizeCols + j];
            }
            if (i == sizeRows - 1) {
                G[i * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[i * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            if (j == sizeCols - 1) {
                G[(i + 1) * sizeCols + j] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j] = theta[i * sizeCols + j];
            }
            // setting the corners
            if (i == 1 && j == 1) {
                G[(i - 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            if (i == 1 && j == sizeCols - 1) {
                G[(i - 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i - 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }
            if (i == sizeRows - 1 && j == 1) {
                G[(i + 1) * sizeCols + j - 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j - 1] = theta[i * sizeCols + j];
            }
            if (i == sizeRows - 1 && j == sizeCols - 1) {
                G[(i + 1) * sizeCols + j + 1] = G[i * sizeCols + j];
                theta[(i + 1) * sizeCols + j + 1] = theta[i * sizeCols + j];
            }

            // round to the nearest 45 degrees
            if (theta[i * sizeCols + j] >= 23 && theta[i * sizeCols + j] < 68) {
                direction[i * sizeCols + j] = 45;
            } else if (theta[i * sizeCols + j] >= 68 && theta[i * sizeCols + j] < 113) {
                direction[i * sizeCols + j] = 90;
            } else if (theta[i * sizeCols + j] >= 113 && theta[i * sizeCols + j] < 157) {
                direction[i * sizeCols + j] = 135;
            } else if (theta[i * sizeCols + j] >= 157 && theta[i * sizeCols + j] < 203) {
                direction[i * sizeCols + j] = 180;
            } else if (theta[i * sizeCols + j] >= 203 && theta[i * sizeCols + j] < 247) {
                direction[i * sizeCols + j] = 225;
            } else if (theta[i * sizeCols + j] >= 247 && theta[i * sizeCols + j] < 293) {
                direction[i * sizeCols + j] = 270;
            } else if (theta[i * sizeCols + j] >= 293 && theta[i * sizeCols + j] < 337) {
                direction[i * sizeCols + j] = 315;
            } else {
                direction[i * sizeCols + j] = 0;
            }
        }
    }
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
        }
    }

    // non-maximum suppression
    for (int i = 1; i < sizeRows - 1; i++) {
        for (int j = 1; j < sizeCols - 1; j++) {
            if (direction[i * sizeCols + j] == 0 || direction[i * sizeCols + j] == 180) {
                if (G[i * sizeCols + j] < G[i * sizeCols + j - 1] || G[i * sizeCols + j] < G[i * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            }
            if (direction[i * sizeCols + j] == 45 || direction[i * sizeCols + j] == 225) {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j + 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j - 1]) {
                    G[i * sizeCols + j] = 0;
                }
            }
            if (direction[i * sizeCols + j] == 90 || direction[i * sizeCols + j] == 270) {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j]) {
                    G[i * sizeCols + j] = 0;
                }
            }
            if (direction[i * sizeCols + j] == 135 || direction[i * sizeCols + j] == 315) {
                if (G[i * sizeCols + j] < G[(i + 1) * sizeCols + j - 1] || G[i * sizeCols + j] < G[(i - 1) * sizeCols + j + 1]) {
                    G[i * sizeCols + j] = 0;
                }
            }

            pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
        }
    }

    // double threshold
    int count = 0;
    int changes = 1;
    while (changes == 1) {
        changes = 0;
        for (int i = 1; i < sizeRows - 1; i++) {
            for (int j = 1; j < sizeCols - 1; j++) {
                if (G[i * sizeCols + j] < (lowerThreshold * largestG)) {
                    G[i * sizeCols + j] = 0;
                } else if (G[i * sizeCols + j] >= (higherThreshold * largestG)) {
                    continue;
                } else if (G[i * sizeCols + j] < (higherThreshold * largestG)) {
                    int tempG = G[i * sizeCols + j];
                    G[i * sizeCols + j] = 0;
                    for (int x = -1; x <= 1; x++) {
                        for (int y = -1; y <= 1; y++) {
                            if (x == 0 && y == 0) { continue; }
                            if (G[(i + x) * sizeCols + (j + y)] >= (higherThreshold * largestG)) {
                                G[i * sizeCols + j] = (higherThreshold * largestG);
                                changes = 1;
                                break;
                            }
                        }
                    }
                }
                pixelsCanny[i * sizeCols + j] = (int)(G[i * sizeCols + j] * (255.0 / largestG));
            }
        }
        count++;
        if (count > (sizeRows * sizeCols)) {
            std::cout << "The count was exceeded.\n";
            return NULL;
        }
    }
    // std::cout << count << "\n";

    return pixelsCanny;
}

// int main() {

//     std::string data;

//     // open a jpeg file
//     // CImage for a library
//     // bitmap might be easier as no compression if no library

//     std::string path = "GokuBlack.bmp";
//     std::fstream file;
//     file.open(path, std::ios_base::out | std::ios::binary);
//     // file >> data;

//     // file.rdbuf();

//     if (file.is_open()) {
//         std::cout << "The file is open." << std::endl;
//         // std::cout << file.rdbuf() << std::endl;
//         // file.read();
//     }
//     file.close();

//     // apply a gausian filter/blur

//     // apply canny edge detection

// }
