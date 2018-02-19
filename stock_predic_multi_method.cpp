//
//  stock_predic_multi_method.cpp
//
//  Created by WeiXu on 12/1/17.
//  Copyright © 2017 WeiXu. All rights reserved.
//


#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

const string winName = "points";

Mat img, imgDst;
RNG rng;

vector<Point>  trainedPoints;
vector<int>    trainedPointsMarkers;
const int MAX_CLASSES = 2;
vector<Vec3b>  classColors(MAX_CLASSES);
int currentClass = 2;
vector<int> classCounters(MAX_CLASSES);

// load training data from a CSV file
/*
 sample variable and their representive feature:
 1: VOLUME * 100 (M),
 2: AVG.VOL(3m) * 100 (M),
 3: Avg.Vol(10d) * 100 (M),
 4: day high * 100,
 5: OPEN * 100 ,
 6: pre-Mkt%Chg * 100 + 10000,
 7: 52-wk high * 100,
 8: 52-wk low * 100,
 9: 1y Target Est * 100,
 10: market cap * 100 (B),
 */
static Ptr<TrainData> prepare_train_data()
{
    Ptr<TrainData> tdata = TrainData::loadFromCSV("data_stock.txt", 0, -1, -1,
                                                  "cat[0-10]");
    //set the train/test split ratio to be 4:1
    tdata->setTrainTestSplitRatio(0.80);
    
    //print the train samples.
    Mat data = tdata->getTrainSamples();

    return tdata;
}

static Mat prepare_train_data_ANN()
{
    Ptr<TrainData> tdata = TrainData::loadFromCSV("data_stock.txt", 0, -1, -1,
                                                  "cat[0-10]");
    //set the train/test split ratio to be 4:1
    tdata->setTrainTestSplitRatio(0.80);
    
    //print the train samples.
    Mat data = tdata->getTrainSamples();
    
    return data;
}


// generate data to test model and run the classifier
static void predict_and_paint_ANN(const Ptr<StatModel>& model,const Mat&  layer_sizes)
{
    Ptr<TrainData> tdata = TrainData::loadFromCSV("data_stock.txt", 0, -1, -1,
                                                  "cat[0-10]");
    tdata->setTrainTestSplitRatio(0.80);
    
    //print test samples
    Mat data_test = tdata->getTestSamples();
    
    //calculate error
    cout << " Hidden layers is " << layer_sizes.cols - 2 << ", and nodes per layers is "<< layer_sizes.at<int>(1)<<endl;
    cout << " calc error :"<< model->calcError( tdata, false, noArray() ) << endl;
    cout << endl;
}

static void find_decision_boundary_ANN(const Mat&  layer_sizes)
{
    Mat samples = prepare_train_data_ANN();
    
    Ptr<TrainData> tdata = prepare_train_data();
    
    Ptr<ANN_MLP> ann = ANN_MLP::create();
    
    /**  Integer vector specifying the number of neurons in each layer including the input and output layers.
     The very first element specifies the number of elements in the input layer.
     The last element - number of elements in the output layer. Default value is empty Mat.
     @sa getLayerSizes */
    ann->setLayerSizes(layer_sizes);
    
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    
    /** Termination criteria of the training algorithm.
     You can specify the maximum number of iterations (maxCount) and/or how much the error could
     change between the iterations to make the algorithm continue (epsilon). Default value is
     TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01).*/
    /** @see setTermCriteria */
    ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, FLT_EPSILON));
    
    /** Sets training method and common parameters.
     @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
     @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP
     @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP.
     */
    ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1);

    Mat labels = tdata->getTrainResponses();
    Mat responses(tdata->getNTrainSamples(), 2, CV_32F, 0.0f);
    for (size_t i=0; i<tdata->getNTrainSamples(); i++) {
        int id = (int)labels.at<float>(i);  // 0 or 1
        responses.at<float>(i, id) = 1;
    }
    ann->train(tdata->getTrainSamples(), 0, responses);

    predict_and_paint_ANN(ann,layer_sizes);
}

// generate data to test model and run the classifier
static void predict_and_paint(const Ptr<StatModel>& model)
{
    Ptr<TrainData> tdata = TrainData::loadFromCSV("data_stock.txt", 0, -1, -1,
                                                  "cat[0-10]");
    tdata->setTrainTestSplit(20,true);
    
    //print test samples
    Mat data_test = tdata->getTestSamples();

    //calculate error
    cout << " calc error of DTree : " << model->calcError( tdata, false, noArray() ) << endl;
    cout << endl;
}
// generate decision tree model and try it
static void find_decision_boundary_DT()
{
    Ptr<DTrees> dtree = DTrees::create();
    dtree->setMaxDepth(8);
    dtree->setMinSampleCount(1);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0); // the number of cross-validation folds
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(false);
    dtree->train(prepare_train_data());
    
    predict_and_paint(dtree);
}
int main()
{
    

    //Hidden layers is 2, and nodes per layers is 3
    Mat layer_sizes1(1, 4, CV_32SC1);
    layer_sizes1.at<int>(0) = 10;
    layer_sizes1.at<int>(1) = 3;
    layer_sizes1.at<int>(2) = 3;
    layer_sizes1.at<int>(3) = 2;
    find_decision_boundary_ANN(layer_sizes1);

    //Hidden layers is 2, and nodes per layers is 6
    Mat layer_sizes2(1, 4, CV_32SC1);
    layer_sizes2.at<int>(0) = 10;
    layer_sizes2.at<int>(1) = 4;
    layer_sizes2.at<int>(2) = 4;
    layer_sizes2.at<int>(3) = 2;
    find_decision_boundary_ANN(layer_sizes2);


    //Hidden layers is 4, and nodes per layers is 3
    Mat layer_sizes3(1, 6, CV_32SC1);
    layer_sizes3.at<int>(0) = 10;
    layer_sizes3.at<int>(1) = 3;
    layer_sizes3.at<int>(2) = 3;
    layer_sizes3.at<int>(3) = 3;
    layer_sizes3.at<int>(4) = 3;
    layer_sizes3.at<int>(5) = 2;
    find_decision_boundary_ANN(layer_sizes3);

    //Hidden layers is 4, and nodes per layers is 6
    Mat layer_sizes4(1, 6, CV_32SC1);
    layer_sizes4.at<int>(0) = 10;
    layer_sizes4.at<int>(1) = 6;
    layer_sizes4.at<int>(2) = 6;
    layer_sizes4.at<int>(3) = 6;
    layer_sizes4.at<int>(4) = 6;
    layer_sizes4.at<int>(5) = 2;
    find_decision_boundary_ANN(layer_sizes4);

    find_decision_boundary_DT();
    return 0;
}




