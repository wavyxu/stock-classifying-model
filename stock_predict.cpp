//
//  stock_predict.cpp
//
//  Created by WeiXu on 11/22/17.
//  Copyright © 2017 WeiXu. All rights reserved.

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

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
    
    cout << "Train Samples : " << endl;
    for (int row = 0; row < data.rows; row++) {
        for (int col = 0; col < data.cols; col++)
            cout << data.at<float>(row, col) << " ";
        cout << endl;
    }

    return tdata;
}

// generate data to test model and run the classifier
static void predict_and_paint(const Ptr<StatModel>& model)
{
    Ptr<TrainData> tdata = TrainData::loadFromCSV("data_stock.txt", 0, -1, -1,
                                                  "cat[0-10]");
    tdata->setTrainTestSplitRatio(0.80);
    
    //print test samples
    Mat data_test = tdata->getTestSamples();
    
    cout << endl;
    cout << "Test Samples : " << endl;
    for (int row = 0; row < data_test.rows; row++) {
        for (int col = 0; col < data_test.cols; col++)
            cout << data_test.at<float>(row, col) << " ";
        cout << endl;
    }
    cout << endl;

    //print predict results. 1: buy,  0: don't buy
    for (int row = 0; row < data_test.rows; row++) {
         cout << "predict of test sampe " <<row <<" : " << model->predict(data_test.row(row),noArray())<<endl;

     }
    cout << endl;
    
    //calculate error
    cout << "calc error: " << model->calcError( tdata, false, noArray() ) << endl;
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
    find_decision_boundary_DT();
    
    system("pause");
    
    return 0;
}



