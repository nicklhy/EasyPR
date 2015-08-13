#include "easypr.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if(argc!=3) {
        cout << "Usage: " << argv[0] << " image res-path" << endl;
        return -1;
    }

    vector<string> license;
    vector<easypr::CPlate> plates;
    string pr_models_dir = "/home/lhy/Documents/Codes/CV/recognition/EasyPR/resources/model";
    easypr::CPlateRecognize pr;
    pr.LoadANN(pr_models_dir+"/ann.xml");
    pr.LoadSVM(pr_models_dir+"/svm.xml");
    pr.setLifemode(true);
    pr.setDebug(false);

    cout << "open image " << argv[1] << endl;
    Mat img = imread(argv[1]);
    assert(img.cols!=0 && img.rows!=0);
    pr.plateRecognize(img, license, plates);

    ofstream out(argv[2]);
    for(size_t i=0; i<license.size(); ++i)
        out << license[i] << endl;
    out.close();

    return 0;
}
