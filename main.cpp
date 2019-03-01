#include <iostream>
#include <fstream>
#include <random>
#include "opencv2/opencv.hpp"
#include "posemapper.h"
#include "lbs.h"

using namespace cv;

struct smpl_model
{
    Mat model;
    Mat J_regressor_prior;
    Mat pose;
    Mat f;
    Mat J_regressor;
    Mat betas;
    Mat kintree_table;
    Mat J;
    Mat v_shaped;
    Mat weights_prior;
    Mat trans;
    Mat v_posed;
    Mat weights;
    Mat vert_sym_idxs;
    Mat posedirs;
    Mat v_template;
    Mat shapedirs;
};

int main() {
    //std::cout << "Hello, World!" << std::endl;
    cv::FileStorage fs("data/data.yml", cv::FileStorage::READ);
    cv::Mat J_regressor_prior, f, J_regressor, kintree_table, J, weights_prior, weights, vert_sym_idxs, posedirs,
            v_template, shapedirs, trans, pose, betas;

    fs["J_regressor_prior"] >> J_regressor_prior;
    fs["f"] >> f;
    fs["J_regressor"] >> J_regressor;
    fs["kintree_table"] >> kintree_table;
    fs["J"] >> J;
    fs["weights_prior"] >> weights_prior;
    fs["weights"] >> weights;
    fs["vert_sym_idxs"] >> vert_sym_idxs;
    fs["posedirs"] >> posedirs;
    fs["v_template"] >> v_template;
    fs["shape"] >> shapedirs;

//    std::cout << "J_regressor_prior: " << J_regressor_prior.size << " and channels: " << J_regressor_prior.channels() << std::endl;
//    std::cout << "f: " << f.size << " and channels: " << f.channels() << std::endl;
//    std::cout << "J_regressor: " << J_regressor.size << " and channels: " << J_regressor.channels() << std::endl;
//    std::cout << "kintree_table: " << kintree_table.size << " and channels: " << kintree_table.channels() << std::endl;
//    std::cout << "J: " << J.size << " and channels: " << J.channels() << std::endl;
//    std::cout << "weights_prior: " << weights_prior.size << " and channels: " << weights_prior.channels() << std::endl;
//    std::cout << "weights: " << weights.size << " and channels: " << weights.channels() << std::endl;
//    std::cout << "vert_sym_idxs: " << vert_sym_idxs.size << " and channels: " << vert_sym_idxs.channels() << std::endl;
//    std::cout << "posedirs: " << posedirs.size << " and channels: " << posedirs.channels() << std::endl;
//    std::cout << "v_template: " << v_template.size << " and channels: " << v_template.channels() << std::endl;
//    std::cout << "shape: " << shapedirs.size << " and channels: " << shapedirs.channels() << std::endl;

    bool want_shapemodel = !shapedirs.empty();
    int nposeprams = kintree_table.cols;

    if (trans.empty()) {
        trans = Mat::zeros(3, 1, CV_64F);
    }
    if (pose.empty()) {
        pose = Mat::zeros(nposeprams * 3, 1, CV_64F);
    }
    if (!shapedirs.empty() && betas.empty()) {
        betas = Mat::zeros(shapedirs.channels(), 1, CV_64F);
    }

    //random pose and betas
    /*std::cout << "----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "prior_model.pose: " << model.pose << std::endl;
    std::cout << "proir_model.betas: " << model.betas << std::endl;*/

    for (size_t i = 0; i < pose.rows; i++)
    {
        pose.at<double>(i, 0) = ((double)rand() / (RAND_MAX)) * 0.2;
//        pose.at<double>(i, 0) = 0.1;
    }
    std::cout << pose << std::endl;
    for (size_t i = 0; i < betas.rows; i++)
    {
        betas.at<double>(i, 0) = ((double)rand() / (RAND_MAX)) * 0.03;
    }

    /*std::cout << "after_model.pose: " << model.pose << std::endl;
    std::cout << "after_model.betas: " << model.betas << std::endl;*/

    Mat v_shaped, v_posed;

    posedirs = posedirs.reshape(1, posedirs.rows * posedirs.cols);
    //std::cout << posedirs.size << std::endl;
    //std::cout << beta << std::endl;

    v_template = v_template.reshape(1, v_template.rows * v_template.cols);
    //std::cout << v_template.size << std::endl;
    shapedirs = shapedirs.reshape(1, shapedirs.rows * shapedirs.cols);
    //std::cout << shapedirs.size << std::endl;

    if (want_shapemodel) {
//        std::cout << "!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        v_shaped = v_template + shapedirs * betas;

        v_shaped = v_shaped.reshape(1, v_shaped.rows / 3);

        Mat Jtmpx, Jtmpy, Jtmpz, tmp;

        //std::cout << v_shaped.size << std::endl;

        Jtmpx = J_regressor * v_shaped.col(0);
        //std::cout << Jtmpx << std::endl;
        //std::cout << Jtmpx.size << std::endl;
        Jtmpy = J_regressor * v_shaped.col(1);
        Jtmpz = J_regressor * v_shaped.col(2);
        hconcat(Jtmpx, Jtmpy, tmp);
        hconcat(tmp, Jtmpz, tmp);
        //std::cout << tmp.size << std::endl;
        J = tmp;    //update J matrix
        //todo update v_shape by using pose mapper
        v_posed = posedirs * posemap(pose); //20670x207 * 207x1
        v_posed = v_posed.reshape(1, v_posed.rows / 3) + v_shaped;  //reshape to 6890x3 + 6890x3
    }
    else {
        //todo update v_shape by using pose mapper
        v_posed = posedirs * posemap(pose); //20670x207 * 207x1
        v_posed = v_posed.reshape(1, v_posed.rows / 3) + v_template.reshape(1, v_template.rows / 3);  //reshape to 6890x3 + 6890x3
    }

//    std::cout << "weight: " << weights.size << " and channels: " << weights.channels() << std::endl;

//    std::cout << v_posed.row(0) << std::endl;

    v_and_J vj = verts_core(pose, v_posed, J, weights, kintree_table, true);

//    std::cout << vj.v.row(0) << std::endl;
//    std::cout << "prior_v: " << vj.v.size << "      prior_Jtr: " << vj.Jtr.size << std::endl;

//    std::cout << "trans: " << trans << std::endl;

    for (size_t i = 0; i < vj.v.rows; i++)
    {
        vj.v.row(i) += trans.t();
    }

    for (size_t i = 0; i < vj.Jtr.rows; i++)
    {
        vj.Jtr.row(i) += trans.t();
    }
    //std::cout << "after_v: " << vj.v.size << "      after_Jtr: " << vj.Jtr.size << std::endl;
    //vj.v += trans;

    /*std::cout << "J_regressor_prior: " << J_regressor_prior.size << std::endl;
    std::cout << "f: " << f.size << std::endl;
    std::cout << "J_regressor: " << J_regressor.size << std::endl;
    std::cout << "kintree_table: " << kintree_table.size << std::endl;
    std::cout << "J: " << J.size << std::endl;
    std::cout << "weights_prior: " << weights_prior.size << std::endl;
    std::cout << "weights: " << weights.size << std::endl;
    std::cout << "vert_sym_idxs: " << vert_sym_idxs.size << std::endl;
    std::cout << "posedirs: " << posedirs.size << std::endl;
    std::cout << "v_template: " << v_template.size << std::endl;
    std::cout << "shape: " << shapedirs.size << " with " << shapedirs.channels() << " channels." << std::endl;*/

    smpl_model model;
    model.model = vj.v;
    model.J_regressor_prior = J_regressor_prior;
    model.pose = pose;
    model.f = f;
    model.J_regressor = J_regressor;
    model.betas = betas;
    model.kintree_table = kintree_table;
    model.J = J;
    model.v_shaped = v_shaped;
    model.weights_prior = weights_prior;
    model.trans = trans;
    model.v_posed = v_posed;
    model.weights = weights;
    model.vert_sym_idxs = vert_sym_idxs;
    model.posedirs = posedirs;
    model.v_template = v_template;
    model.shapedirs = shapedirs;

//    std::cout << pose << std::endl;
//    std::cout << "----------------------------------------------------------------------------------------------------------" << std::endl;
//    std::cout << "model: " << model.model << std::endl;
//    std::cout << "J_regressor_prior: " << model.J_regressor_prior.size << std::endl;
//    std::cout << "pose: " << model.pose.size << std::endl;
//    std::cout << "f: " << model.f.size << std::endl;
//    std::cout << "J_regressor: " << model.J_regressor.size << std::endl;
//    std::cout << "betas: " << model.betas.size << std::endl;
//    std::cout << "kintree_table: " << model.kintree_table.size << std::endl;
//    std::cout << "J: " << model.J.size << std::endl;
//    std::cout << "v_shaped: " << model.v_shaped.size << std::endl;
//    std::cout << "weights_prior: " << model.weights_prior.size << std::endl;
//    std::cout << "trans: " << model.trans.size << std::endl;
//    std::cout << "v_posed: " << model.v_posed.size << std::endl;
//    std::cout << "weights: " << model.weights.size << std::endl;
//    std::cout << "vert_sym_idxs: " << model.vert_sym_idxs.size << std::endl;
//    std::cout << "posedirs: " << model.posedirs.size << std::endl;
//    std::cout << "v_template: " << model.v_template.size << std::endl;
//    std::cout << "shapedirs: " << model.shapedirs.size << std::endl;

    std::cout << "----------------------------------------------------------------------------------------------------------" << std::endl;

    std::ofstream meshfile ("SMPL_Mesh.obj");
    if (meshfile.is_open())
    {
        for (size_t i = 0; i < model.model.rows; i++)
        {
            meshfile << "v " << model.model.at<double>(i, 0) << " " << model.model.at<double>(i, 1) << " " << model.model.at<double>(i, 2) << "\n";
        }
        for (size_t i = 0; i < model.f.rows; i++)
        {
            meshfile << "f " << model.f.at<double>(i, 0) + 1 << " " << model.f.at<double>(i, 1) + 1 << " " << model.f.at<double>(i, 2) + 1 << "\n";
        }
        std::cout << "Finish writing SMPL Mesh!" << std::endl;
    }
    else std::cout << "Unable to open file";

    return 0;
}