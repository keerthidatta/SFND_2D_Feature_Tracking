/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <numeric>
#include <map>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../../";
    
    // Open the file
    std::ifstream file(dataPath + "params/parameters.txt");

    // Check if the file is opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        return 1;
    }

    string detectorType = "SIFT"; //HARRIS, FAST
    string descriptorType = "SIFT"; // BRIEF, ORB, FREAK, AKAZE, SIFT
    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string descriptorNorm = "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    string parameterName;
    string parameterValue;

    while (file >> parameterName >> parameterValue) 
    {
        if(parameterName == "detectorType")
        {
            detectorType = parameterValue;
        }
        else if(parameterName == "descriptorType")
        {
            descriptorType = parameterValue;
        }
        else if(parameterName == "matcherType")
        {
            matcherType = parameterValue;
        }
        else if(parameterName == "descriptorNorm")
        {
            descriptorNorm = parameterValue;
        }
        else if(parameterName == "selectorType")
        {
            selectorType = parameterValue;
        }
    }

    // Close the file
    file.close();

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 3;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        frame.imName = imgNumber.str();
        //int framePostion = imgIndex % dataBufferSize;
        
        if(dataBuffer.size() == dataBufferSize)
        {
            auto it = dataBuffer.begin();
            dataBuffer.erase(it);
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if(detectorType.compare("HARRIS")==0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if(detectorType.compare("FAST")==0)
        {
            detKeypointsFast(keypoints, imgGray, false);
        }
        else if(detectorType.compare("BRISK")==0)
        {
            detKeypointsBrisk(keypoints, imgGray, false);
        }
        else if(detectorType.compare("ORB")==0)
        {
            detKeypointsOrb(keypoints, imgGray, false);
        }        
        else if(detectorType.compare("AKAZE")==0)
        {
            detKeypointsAkaze(keypoints, imgGray, false);
        }
        else if(detectorType.compare("SIFT")==0)
        {
            detKeypointsSift(keypoints, imgGray, false);
        }
        else
        {
            std::cout << "Not implemented" << std::endl;
            break;
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        bool bVis = false;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            std::cout << "*** size of keypoints IN Image: " << keypoints.size() << std::endl;

            auto isKeypointOutsideROI = [&vehicleRect](const cv::KeyPoint &keypoint)
            {
                double x = keypoint.pt.x;
                double y = keypoint.pt.y;
                return !((x >= vehicleRect.x) && (x <= (vehicleRect.x + vehicleRect.height)) && (y >= vehicleRect.y) && (y <= (vehicleRect.y + vehicleRect.width)));
            };

            keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), isKeypointOutsideROI), keypoints.end());

            std::cout << "*** size of keypoints IN ROI: " << keypoints.size() << std::endl;
            // visualize results
            if (bVis)
            {
                cv::Mat visImage = imgGray.clone();
                cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                string windowName = "filtered keypoints";
                cv::namedWindow(windowName, 6);
                imshow(windowName, visImage);
                cv::waitKey(0);
            }
        }
        bool bStatistics = true;

        if(bStatistics)
        {
            if(detectorType.compare("SIFT")==0 || detectorType.compare("BRISK")==0)
            {
                std::vector<float> neighborhoodSizes;
                for(const auto &keypoint:keypoints)
                {
                    float size = keypoint.size;
                    neighborhoodSizes.push_back(size);
                }
                double sum = std::accumulate(neighborhoodSizes.begin(), neighborhoodSizes.end(), 0.0);
                double mean = sum/neighborhoodSizes.size();
                std::cout << "*** Average neighborhood size is: " << mean << std::endl;
            }
        }
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;

        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;

        if(descriptorType == "AKAZE" && detectorType != "AKAZE")
        {
            std::cout << "AKAZE descriptor works only with Akaze keypoint detection" << std::endl;
            break;
        }
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorNorm, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done : # matches" << matches.size() << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}