//
// Created by Vladislav Tyurin on 04.12.17.
//
#include "ImageTransformer.h"

void ImageTransformer::walk()
{
    using directory_iterator = boost::filesystem::recursive_directory_iterator;
    assert(boost::filesystem::exists(m_DataSetDir));

    for(auto it = directory_iterator(m_DataSetDir), end = directory_iterator(); it!=end;++it)
    {
        if (it->path().filename() == ".DS_Store")
            continue;

        if (boost::filesystem::is_directory(it->path()))
            continue;

        augmentImage(it->path().string());
    }
}

void ImageTransformer::augmentImage(const std::string & filepath)
{
    m_Image = cv::imread(filepath, CV_LOAD_IMAGE_COLOR);
    auto fileName = boost::filesystem::path(filepath).filename();
    auto directoryName = boost::filesystem::path(filepath).remove_filename().leaf();
    std::string subDir = m_DestDir+std::to_string(m_NewSize);
    if(!boost::filesystem::exists(subDir))
        boost::filesystem::create_directory(subDir);

    subDir+="/";
    subDir+=directoryName.generic_string();
    if(!boost::filesystem::exists(subDir))
        boost::filesystem::create_directory(subDir);

    auto extension = fileName.extension();
    fileName.replace_extension("");

    if(!m_Image.data)
    {
        return;
    }


    cv::Mat newImage;
    cv::Rect r;
    std::string newFileName = subDir;
    newFileName += "/";

    int k =2; //scale factor
    int regionSize = m_Image.size().height;
    while (regionSize<=m_NewSize)
    {
        cv::resize(m_Image,m_Image,cv::Size(regionSize,regionSize),0,0,CV_INTER_CUBIC);
        for (int i = 0; i < m_NewSize; i += regionSize)
        {
            for (int j = 0; j < m_NewSize; j += regionSize)
            {
                newImage = cv::Mat(cv::Size(m_NewSize,m_NewSize),CV_8UC3,cv::Scalar::all(0)); // 8-битное одноканальное изображение;
                r = cv::Rect(i,j,regionSize,regionSize);
                newImage(r) += m_Image;

                std::string newFileName = subDir;
                newFileName += "/";
                newFileName += fileName.generic_string();
                newFileName += "_";
                newFileName += std::to_string(regionSize);
                newFileName += std::to_string(i);
                newFileName += std::to_string(j);
                newFileName += extension.generic_string();

                std::cout<<newFileName<<"\n";
                cv::imwrite(newFileName,newImage);
            }
        }
        regionSize*=k;
    }
}

