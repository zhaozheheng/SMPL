//
// Created by Zheheng Zhao on 11/4/18.
//

#ifndef SMPL_LBS_H
#define SMPL_LBS_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "posemapper.h"

using namespace cv;

struct rigid
{
    Mat result;
    std::vector<Mat> results_global;
};

struct v_and_J
{
    Mat v;
    Mat Jtr;
};

Mat with_zeros(Mat theta, Mat joint) {
    Mat tmp, result;
    Mat rod;
    double data[4] = { 0.0, 0.0, 0.0, 1.0 };
    cv::Rodrigues(theta, rod);
    //std::cout << rod << std::endl;
    //std::cout << joint << std::endl;
    hconcat(rod, joint.t(), tmp);
    //std::cout << tmp << std::endl;
    vconcat(tmp, Mat(1, 4, CV_64F, data), result);
    //std::cout << result << std::endl;
    return result;
}

Mat pack(Mat mat) {
    Mat result;
    Mat tmp = Mat::zeros(4, 3, CV_64F);
    hconcat(tmp, mat, result);
    return result;
}

rigid global_rigid_transformation(Mat pose, Mat J, Mat kintree_table) {
    //std::cout << "pose: " << pose.size << std::endl;
    //std::cout << "J: " << J.size << std::endl;
    //std::cout << "kintree_table: " << kintree_table.size << std::endl;

    std::vector<Mat> results(kintree_table.cols);
    std::vector<Mat> results_global;
    Mat result = Mat::zeros(16, 0, CV_64F);
    pose = pose.reshape(1, pose.rows / 3);	//24x3
    //std::cout << "pose: " << pose.size << std::endl;
    std::vector<double> id_to_col, parent;
    for (int i = 0; i < kintree_table.cols; ++i) {
        id_to_col.push_back(kintree_table.at<double>(1, i));
    }

    /*for (int i = 0; i < id_to_col.size(); ++i) {
        std::cout << id_to_col[i] << " ";
    }
    std::cout << std::endl;*/

    parent.push_back(0.0);
    for (int i = 1; i < kintree_table.cols; ++i) {  //start from index 1, since index 0 is root
        parent.push_back(kintree_table.at<double>(0, i));
    }

    /*for (int i = 0; i < parent.size(); ++i) {
        std::cout << "parent " << i << ": " << parent[i] << std::endl;
    }*/

    //initialize the 1st result
    results[0] = with_zeros(pose.row(0), J.row(0));

    //todo continue compute the following results

    //std::cout << "results[0]: " << results[0] << std::endl;

    //std::cout << "OK!" << std::endl;

    for (int i = 1; i < kintree_table.cols; ++i) {  //start from index 1, since index 0 is root
        results[i] = results[parent[i]] * with_zeros(pose.row(i), J.row(i) - J.row(parent[i]));
        //std::cout << results[i] << std::endl;
    }

    results_global = results;

    if (1) {
        std::vector<Mat> results2(kintree_table.cols);
        for (int i = 0; i < results.size(); ++i) {
            double data[1] = { 0.0 };
            Mat tmp;
            //std::cout << results[i].size << std::endl;
            //std::cout << J.row(i) << std::endl;
            hconcat(J.row(i), Mat{ 1, 1, CV_64F, data }, tmp);
            //std::cout << tmp << std::endl;
            //std::cout << tmp.size << std::endl;
            results2[i] = results[i] - pack(results[i] * tmp.t());
        }
        results = results2;
    }

//    for (int i = 0; i < results.size(); ++i) {
//        std::cout << results[i] << std::endl;
//    }


    for (int i = 0; i < results.size(); ++i) {
        results[i] = results[i].reshape(1, results[i].rows * results[i].cols);
        //std::cout << result.size << std::endl;
        //std::cout << results[i].size << std::endl;
        hconcat(result, results[i], result);
        //std::cout << result << std::endl;
    }
//    std::cout << result << std::endl;
//    for (int j = 0; j < result.rows; ++j) {
//        std::cout << result.t().row(j) << std::endl;
//    }
//    std::cout << result.row(0) << std::endl;
//    std::cout << results_global.size() << std::endl;
//    std::cout << results_global[0] <<std::endl;

    rigid rt = { result, results_global };
    return rt;
}

v_and_J verts_core(Mat pose, Mat v, Mat J, Mat weights, Mat kintree_table, bool want_Jtr = false) {
    /*std::cout << "pose: " << pose.size << std::endl;
    std::cout << "v: " << v.size << std::endl;
    std::cout << "J: " << J.size << std::endl;
    std::cout << "weights: " << weights.size << std::endl;
    std::cout << "kintree_table: " << kintree_table.size << std::endl;*/

    v_and_J vj;
    rigid rt = global_rigid_transformation(pose, J, kintree_table);
    Mat T, rest_shape_h;
//    std::cout << weights.row(0) << std::endl;
    T = rt.result * weights.t();
    vconcat(v.t(), Mat::ones(1, v.rows, CV_64F), rest_shape_h);
//    std::cout << rest_shape_h <<std::endl;
//    std::cout << rt.result.col(0) <<std::endl;
//    std::cout << T.col(0) <<std::endl;
    //std::cout << "T: " << T.size << std::endl;
    //std::cout << "rest_shape_h: " << rest_shape_h.size << std::endl;
    //std::cout << "v: " << v.size << std::endl;
//    std::cout << T.rowRange(0, 4).col(0) << std::endl;
//    std::cout << rest_shape_h.row(0) << std::endl;
    //todo problem here
//    std::cout << T.row(0) << std::endl;
//    std::cout << T.row(4) << std::endl;
//    std::cout << T.row(8) << std::endl;
//    std::cout << T.row(12) << std::endl;
//    std::cout << rest_shape_h.row(0) << std::endl;
//    std::cout << T.at<double>(0, 0) << std::endl;
//    Mat tmp = (T.rowRange(0, 4).col(0) * rest_shape_h.row(0) + T.rowRange(4, 8).col(0) * rest_shape_h.row(1) + T.rowRange(8, 12).col(0) * rest_shape_h.row(2) + T.rowRange(12, 16).col(0) * rest_shape_h.row(3)).t();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < T.row(i).cols; ++j) {
            T.at<double>(4 * i, j) *= rest_shape_h.row(0).at<double>(0, j);
            T.at<double>(4 * i + 1, j) *= rest_shape_h.row(1).at<double>(0, j);
            T.at<double>(4 * i + 2, j) *= rest_shape_h.row(2).at<double>(0, j);
            T.at<double>(4 * i + 3, j) *= rest_shape_h.row(3).at<double>(0, j);
        }
    }
    //std::cout << T.row(0) << std::endl;
    //std::cout << T.row(4) << std::endl;
    //std::cout << T.row(8) << std::endl;
    //std::cout << T.row(12) << std::endl;

    Mat tmp;

    Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = T.row(0) + T.row(1) + T.row(2) + T.row(3);
    tmp2 = T.row(4) + T.row(5) + T.row(6) + T.row(7);
    tmp3 = T.row(8) + T.row(9) + T.row(10) + T.row(11);
    tmp4 = T.row(12) + T.row(13) + T.row(14) + T.row(15);

    //std::cout << tmp1.size << std::endl;

    vconcat(tmp1, tmp2, tmp);
    vconcat(tmp, tmp3, tmp);
    vconcat(tmp, tmp4, tmp);
    std::cout << tmp.size << std::endl;
    tmp = tmp.t();
    std::cout << "tmp: " << tmp << std::endl;
//    std::cout << T.rowRange(0, 4) << std::endl;
//    std::cout << T.rowRange(4, 8).col(0) * rest_shape_h.row(1) << std::endl;
//    std::cout << T.rowRange(0, 4).col(0) * rest_shape_h.row(0) + T.rowRange(4, 8).col(0) * rest_shape_h.row(1) << std::endl;
//    std::cout << tmp.col(0) << std::endl;
    vj.v = tmp.colRange(0, 3);
//    std::cout << "vj.v: " << vj.v << std::endl;
//    std::cout << "rt.results_global: " << rt.results_global[0].size << std::endl;

    Mat Jtr = rt.results_global[0].col(3).rowRange(0, 3).t();

    for (size_t i = 1; i < rt.results_global.size(); i++)
    {
        vconcat(Jtr, rt.results_global[i].col(3).rowRange(0, 3).t(), Jtr);
    }
//    std::cout << "Jtr: " << Jtr << std::endl;
    vj.Jtr = Jtr;
    return vj;
}

#endif //SMPL_LBS_H
