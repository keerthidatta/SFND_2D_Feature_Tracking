# Camera based 2D Feature Tracking Report

## Data Buffer Optimization
A data buffer (std::vector) is implemented in a way that a maximum of dataBufferSize number of elements is stored in databuffer. 
The front most element of the vector is removed before adding a new element to the buffer. 

## Keypoint Detection and Removal

There are multiple keypoint detectors available in opencv. Based on the keypoint detection algorithm string type, 
Keypoints are detected and stord in a vector of type cv::KeyPoint.  

The keypoints outside the region of interest are removed by finding if the keypoint lies inside the rectangle of minimum point (x,y) and maximum point (x, y). 

## Descriptors

* Keypoint descriptors - Find keypoint descriptors for the specified string. Descriptor computation is done using opencv implmentations based on the computed keypoints. 
The descriptors are of type cv::Mat

* FLANN matching - DescriptorMatcher object of FLANNBASED is created and based on the type of matching algorithm, the best match for each descriptor in descriptor source is found in descriptor reference

* Distance Ratio - The goal here is to filer out incorrect matches and improve robustness of feature matching. After KNN, descriptor ratio test is performed to filter out unreliable matches. Based on the two closest matches of a keypoint, distance of the first match is compared with the second one. If the ratio of the distance threshold is less than 0.8*distnce of second match, then th ematch is considered as reliable. Otherwise, it is discarded.
   
## Performance
FeatureTracking.xlsx contains four sheets with various performance metrics.

* Average Number of keypoints detected - please refer to first sheet - The number of keypoints detected and the keypoints in the ROI are noted.
* Distribuitions of the neighborhood size - refer to "Distribution" sheet - Distributions of SIFT and BRISK are noted
* Average Number of keypoints matched - refer to Matched Keypoints sheet. 
* Execution time - All detector descriptor combinations are documented in Execution time sheet. It can be seen that FAST detector is at least 20 times faster than all the other detectors

* Based on the performance evaluation of all the detectors and descriptors, best combination is FAST-BRIEF based on the execution time, which takes approximately 3 to 4 mili seconds to detect keypoints and extract descriptor. Matching results are also satisfactory for this combination. The other combination that could work well are FAST-ORB, ORB-BRIEF.