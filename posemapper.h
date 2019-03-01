//
// Created by Zheheng Zhao on 11/2/18.
//

#ifndef SMPL_POSEMAPPER_H
#define SMPL_POSEMAPPER_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

Mat posemap(Mat pose) {
    Mat exp;
    Mat p = pose.rowRange(3, pose.rows).clone();
    p = p.reshape(1, p.rows / 3);
    //std::cout << p.size << std::endl;
    for (int i = 0; i < p.rows; ++i) {
        Mat tmp = p.row(i);
        Mat rod;
        cv::Rodrigues(tmp, rod);
        rod = rod - Mat::eye(3, 3, CV_64F);
        exp.push_back(rod);
        //std::cout << rod << std::endl;
    }
    //std::cout << exp.reshape(1, exp.rows * exp.cols).size;
    return exp.reshape(1, exp.rows * exp.cols);
}

#endif //SMPL_POSEMAPPER_H
