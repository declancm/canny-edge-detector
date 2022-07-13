#include <iostream>

#include "canny.h"

int main() {
    std::string readLocation = "../images/Sukuna.jpg";
    std::string writeLocation = "../images/SukunaCanny.jpg";
    double lowerThreshold = 0.03;
    double higherThreshold = 0.1;

    cannyEdgeDetection(readLocation, writeLocation, lowerThreshold, higherThreshold);

    return 0;
}
