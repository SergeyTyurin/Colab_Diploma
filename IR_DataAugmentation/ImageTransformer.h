//
// Created by Vladislav Tyurin on 04.12.17.
//

#ifndef IR_DATAAUGMENTATION_IMAGETRANSFORMER_H
#define IR_DATAAUGMENTATION_IMAGETRANSFORMER_H

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>

class ImageTransformer{
public:
    ImageTransformer() = default;
    ImageTransformer(int size):m_NewSize(size){};
    void walk();
private:

    void augmentImage(const std::string&);

    cv::Mat m_Image;
    std::string m_DataSetDir = "/home/styurin/Dest/Classification_Images_4";
    std::string m_DestDir = "/home/styurin/Dest/Classification2cl_Images_4_Aug";
    int m_NewSize = 224;
};

#endif //IR_DATAAUGMENTATION_IMAGETRANSFORMER_H
