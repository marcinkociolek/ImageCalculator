#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>

#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NormalizationLib.h"
#include "DispLib.h"
#include "histograms.h"

#include "mazdaroi.h"
#include "mazdaroiio.h"

#include <tiffio.h>

typedef MazdaRoi<unsigned int, 2> MR2DType;

using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          My functions outside the Mainwindow class
//------------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------
Mat LoadROI(boost::filesystem::path InputFile,int maxX, int maxY)
{

    Mat Mask = Mat::zeros(maxY,maxX,CV_16U);
    unsigned short *wMask;

    vector <MR2DType*> ROIVect;

    if(!exists(InputFile))
        return Mask;

    ROIVect = MazdaRoiIO<MR2DType>::Read(InputFile.string());

    unsigned int imSize[2];
    imSize[0] = Mask.rows;
    imSize[1] = Mask.cols;

    int maxXY = maxX*maxY;

    MazdaRoiResizer<MR2DType> resizer;


    int numRois = ROIVect.size();

    if (numRois > 100)
        numRois = 100;


    for(int i = 0; i < numRois;i++)
    {
        if(!ROIVect.at(i)->IsEmpty())
        {
            MR2DType *ROI = resizer.Upsize(ROIVect.at(i),imSize);
            MazdaRoiIterator<MR2DType> iterator(ROI);
            wMask = (unsigned short*)Mask.data;
            while(! iterator.IsBehind())
            {
                if (iterator.GetPixel())
                    *wMask = i+1;
                ++iterator;
                wMask++;
            }
            delete ROI;
        }
    }
    while(ROIVect.size() > 0)
    {
         delete ROIVect.back();
         ROIVect.pop_back();
    }
    return Mask;
}
//------------------------------------------------------------------------------------------------------------------------------
string InterpolationToString(int interpolationNr)
{
    switch(interpolationNr)
    {
    case CV_INTER_NN:
        return "interpolation nearest neighbour";
        break;
    case CV_INTER_LINEAR:
        return "interpolation bilinear";
        break;
    case CV_INTER_CUBIC:
        return "interpolation bicubic";
        break;
    case CV_INTER_AREA:
        return "interpolation area";
        break;
    case CV_INTER_LANCZOS4:
        return "interpolation Lanczos";
        break;
    default:
        return "unrecognized interpolation";
        break;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
/*
string MatPropetiesAsText(Mat Im)
{
    string Out ="Image properties: ";
    Out += "max x = " + to_string(Im.cols);
    Out += ", max y = " + to_string(Im.rows);
    Out += ", # channels = " + to_string(Im.channels());
    Out += ", depth = " + to_string(Im.depth());

    switch(Im.depth())
    {
    case CV_8U:
        Out += "CV_8U";
        break;
    case CV_8S:
        Out += "CV_8S";
        break;
    case CV_16U:
        Out += "CV_16U";
        break;
    case CV_16S:
        Out += "CV_16S";
        break;
    case CV_32S:
        Out += "CV_32S";
        break;
    case CV_32F:
        Out += "CV_32F";
        break;
    case CV_64F:
        Out += "CV_64F";
        break;
    default:
        Out += " unrecognized ";
    break;
    }
    return Out;
}

//------------------------------------------------------------------------------------------------------------------------------
string TiffFilePropetiesAsText(string FileName)
{
    float xRes,yRes;
    uint32 imWidth, imLength;
    uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);


        Out += "max x = " + to_string(imLength);
        Out += ", max y = " + to_string(imWidth);
        Out += ", ResUnit = " + to_string(resolutionUnit);
        Out += ", xRes = " + to_string(1.0/xRes);
        Out += ", yRes = " + to_string(1.0/yRes);
        TIFFClose(tifIm);
    }
    else
        Out += " improper file ";
    //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH, &width);

    return Out;
}
*/
//------------------------------------------------------------------------------------------------------------------------------
bool GetTiffProperties(string FileName, float &xRes, float &yRes)
{
    //float xRes,yRes;
    //uint32 imWidth, imLength;
    //uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        //TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        //TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);

        TIFFClose(tifIm);
        return 1;
    }
    else
    {
        xRes = 1.0;
        yRes = 1.0;
        return 0;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
Mat CreateNormalisedImage16U(Mat ImIn, double minNorm, double maxNorm, int nrOfBins)
{
    Mat ImOut;
    ImOut.release();
    if(ImIn.empty())
        return ImOut;
    if(ImIn.channels() != 1)
        return ImOut;
    if(ImIn.type() != CV_16U)
        return ImOut;

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;
    int maxXY = maxX*maxY;

    if(maxXY == 0)
        return ImOut;
    ImOut = Mat::zeros(maxY,maxX,CV_16U);


    double maxVal = (double)(nrOfBins-1);
    double offset = minNorm;
    double normRange = maxNorm - minNorm;
    if(normRange == 0.0)
        normRange = 1.0;
    double coeff = maxVal/normRange;

    uint16 *wImIn  = (uint16 *)ImIn.data;
    uint16 *wImOut  = (uint16 *)ImOut.data;



    for(int i = 0; i < maxXY; i++)
    {
        double val = ((double)*wImIn - offset) * coeff;
        if(val > maxVal)
            val = maxVal;
        if(val < 0)
            val = 0;
        *wImOut = round(val);

        wImIn++;
        wImOut++;
    }
    return ImOut;
}
//------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------
//          My functions in the Mainwindow class
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ready = 0;
    ui->comboBoxImageInterpolationMethod->addItem(QString::fromStdString(InterpolationToString(0)));
    ui->comboBoxImageInterpolationMethod->addItem(QString::fromStdString(InterpolationToString(1)));
    ui->comboBoxImageInterpolationMethod->addItem(QString::fromStdString(InterpolationToString(2)));
    ui->comboBoxImageInterpolationMethod->addItem(QString::fromStdString(InterpolationToString(3)));
    ui->comboBoxImageInterpolationMethod->addItem(QString::fromStdString(InterpolationToString(4)));

    ui->comboBoxImageInterpolationMethod->setCurrentIndex(CV_INTER_AREA);
    resizeInterpolation = ui->comboBoxImageInterpolationMethod->currentIndex();

    ui->comboBoxDisplayRange->addItem("normal");
    ui->comboBoxDisplayRange->addItem("fixed");
    ui->comboBoxDisplayRange->addItem("minMax");
    ui->comboBoxDisplayRange->addItem("+/-3 sigma");
    ui->comboBoxDisplayRange->addItem("1% - 3%");

    ui->comboBoxDisplayRange->setCurrentIndex(2);

    ui->comboBoxGradientDirection->addItem("X");
    ui->comboBoxGradientDirection->addItem("Y");
    ui->comboBoxGradientDirection->addItem("XY");

    ui->comboBoxROINorm->addItem("min-max");
    ui->comboBoxROINorm->addItem("+/-3 sigma");
    ui->comboBoxROINorm->addItem("1% - 3%");

    ui->comboBoxROINorm->setCurrentIndex(0);

    ui->comboBoxViewROINorm->addItem("min-max");
    ui->comboBoxViewROINorm->addItem("+/-3 sigma");
    ui->comboBoxViewROINorm->addItem("1% - 3%");

    ui->comboBoxViewROINorm->setCurrentIndex(0);



    resizeScale = 0.5;
    ui->lineEditImageScale->setText(QString("%1") .arg(resizeScale));
//    int a0 = CV_INTER_NN;
//    int a1 = CV_INTER_LINEAR;
//    int a2 = CV_INTER_CUBIC;
//    int a3 = CV_INTER_AREA;
//    int a4 = CV_INTER_LANCZOS4;

    operationMode = ui->tabWidgetMode->currentIndex();

    showInImage = ui->checkBoxShowInput->checkState();
    displayScale = pow(double(ui->spinBoxScaleBase->value()), double(ui->spinBoxScalePower->value()));

    ui->textEditOut->clear();

    rngNormalDist = new boost::minstd_rand(time(0));
    normalDistribution = new boost::normal_distribution<>(0.0, 1.0);
    RandomGenNormDistribution = new boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<>>(*rngNormalDist, *normalDistribution);

    rngUniformDist = new boost::minstd_rand(time(0));
    uniformDistribution = new boost::uniform_int<>(ui->spinBoxUniformNoiseStart->value(), ui->spinBoxUniformNoiseStop->value());
    RandomGenUniformDistribution = new boost::variate_generator<boost::minstd_rand&, boost::uniform_int<>>(*rngUniformDist, *uniformDistribution);

    ui->comboBoxRoiShape->addItem("Rectange");
    ui->comboBoxRoiShape->addItem("Circle");

    ui->spinBoxRoiShift->setMinimum( ui->spinBoxRoiSize->value());
    ui->spinBoxRoiOffset->setMinimum( ui->spinBoxRoiSize->value()/2);

    ready = 1;
}
//------------------------------------------------------------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    delete rngNormalDist;
    delete normalDistribution;
    delete RandomGenNormDistribution;

    delete rngUniformDist;
    delete uniformDistribution;
    delete RandomGenUniformDistribution;


    delete ui;
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::OpenImageFolder()
{
    if (!exists(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " not exists "));
        ImageFolder = "d:\\";
    }
    if (!is_directory(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " This is not a directory path "));
        ImageFolder = "C:\\Data\\";
    }
    ui->lineEditImageFolder->setText(QString::fromStdString(ImageFolder.string()));
    ui->listWidgetImageFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(ImageFolder))
    {
        regex FilePattern(ui->lineEditRegexImageFile->text().toStdString());
        if (!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern ))
            continue;
        path PathLocal = FileToProcess.path();
        if (!exists(PathLocal))
        {
            ui->textEditOut->append(QString::fromStdString(PathLocal.filename().string() + " File not exists" ));
            break;
        }
        ui->listWidgetImageFiles->addItem(PathLocal.filename().string().c_str());
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ModeSelect()
{
    if(!ready)
        return;
    ReadImage();
    switch(operationMode)
    {
    case 0:
        TiffRoiFromRed();
        break;
    case 1:
        ImageResize();
        break;
    case 2:
        ImageLinearOperation();
        break;
    case 3:
        CreateROI();
        break;
    case 4:
        OutString += CreateMaZdaScript();
        break;
    case 5:
        ViewRoi();
        break;
    default:

            break;
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ReadImage()
{
    if(ui->checkBoxAutocleanOut->checkState())
        ui->textEditOut->clear();
    int flags;
    if(ui->checkBoxLoadAnydepth->checkState())
        flags = CV_LOAD_IMAGE_ANYDEPTH;
    else
        flags = IMREAD_COLOR;
    ImIn = imread(FileName, flags);
    if(ImIn.empty())
    {
        ui->textEditOut->append("improper file");
        return;
    }

    path FileNamePath(FileName);
    string extension = FileNamePath.extension().string();

    if(extension == ".tif" || extension == ".tiff")
    {
        float xRes, yRes;
        GetTiffProperties(FileName, xRes, yRes);
        xPixelSize = 1.0/(double)xRes;
        if(!ui->checkBoxKeeprequestedPixelSize->checkState())
        {
            xPixSizeOut = xPixelSize / resizeScale;
            ui->lineEditPixelSize->setText(QString::fromStdString(to_string(xPixSizeOut)));
        }
        else
        {
            resizeScale = xPixelSize / xPixSizeOut;
            ui->lineEditImageScale->setText(QString::fromStdString(to_string(resizeScale)));
        }

    }
    else
    {
        xPixelSize = 1.0;
    }

    if(ui->checkBoxShowTiffInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(TiffFilePropetiesAsText(FileName)));

    if(ui->checkBoxShowMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImIn)));

    if(ui->checkBoxShowInput->checkState())
        ShowsScaledImage(ImIn, "Input Image", displayScale);
    if(ui->checkBoxShowInputModyfied->checkState())
        ShowsScaledImage(ImIn, "Input Image PC", displayScale, ui->comboBoxDisplayRange->currentIndex());
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::GetDisplayRange(Mat Im, int dispMode, double *minDisp, double *maxDisp)
{
    switch(dispMode)
    {
    case 1:
        *minDisp = ui->doubleSpinBoxFixMinDisp->value();
        *maxDisp = ui->doubleSpinBoxFixMaxDisp->value();
        break;
    case 2:
        NormParamsMinMax(Im, maxDisp, minDisp);
        break;
    case 3:
        NormParamsMeanP3Std(Im, maxDisp, minDisp);
        break;
    case 4:
        NormParams1to99perc(Im, maxDisp, minDisp);
        break;
    default:
        break;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::GetDisplayRange(Mat Im, Mat Mask, uint16_t RoiNr, int dispMode, double *minDisp, double *maxDisp)
{
    switch(dispMode)
    {
    case 1:
        *minDisp = ui->doubleSpinBoxFixMinDisp->value();
        *maxDisp = ui->doubleSpinBoxFixMaxDisp->value();
        break;
    case 2:
        NormParamsMinMax(Im, Mask, RoiNr, maxDisp, minDisp);
        break;
    case 3:
        NormParamsMeanP3Std(Im, Mask, RoiNr, maxDisp, minDisp);
        break;
    case 4:
        NormParams1to99perc(Im, Mask, RoiNr, maxDisp, minDisp);
        break;
    default:
        break;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName, double dispScale,int dispMode)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }
    Mat ImToShow;

    if(dispMode > 0)
    {
        double minDisp, maxDisp;
        GetDisplayRange(Im, dispMode, &minDisp, &maxDisp);

        Mat ImF;
        Im.convertTo(ImF, CV_64F);
        ImToShow = ShowImageF64PseudoColor(ImF, minDisp, maxDisp);
        ui->textEditOut->append("range " + QString::number(minDisp) + " - " + QString::number(maxDisp));
    }
    else
        ImToShow = Im.clone();

    if (dispScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);

    imshow(ImWindowName, ImToShow);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, Mat Mask, string ImWindowName, double dispScale, uint16_t RoiNr, int dispMode )
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;

    if(dispMode > 0)
    {
        double minDisp, maxDisp;
        GetDisplayRange(Im, Mask, RoiNr, dispMode, &minDisp, &maxDisp);

        Mat ImF;
        Im.convertTo(ImF, CV_64F);
        ImToShow = ShowImageF64PseudoColor(ImF, minDisp, maxDisp);
        ui->textEditOut->append("range " + QString::number(minDisp) + " - " + QString::number(maxDisp));
    }
    else
        ImToShow = Im.clone();

    if (dispScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), dispScale, dispScale, INTER_AREA);
    imshow(ImWindowName, ImToShow);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::SaveScaledImage(Mat Im, string FileName, double dispScale,int dispMode)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;

    if(dispMode > 0)
    {
        double minDisp, maxDisp;
        GetDisplayRange(Im, dispMode, &minDisp, &maxDisp);

        Mat ImF;
        Im.convertTo(ImF, CV_64F);
        ImToShow = ShowImageF64PseudoColor(ImF, minDisp, maxDisp);
    }
    else
        ImToShow = Im.clone();

    if (dispScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    imwrite(FileName, ImToShow);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::SaveScaledImage(Mat Im, Mat Mask, string FileName, double dispScale, uint16_t RoiNr, int dispMode )
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to savew");
        return;
    }

    Mat ImToShow;

    if(dispMode > 0)
    {
        double minDisp, maxDisp;
        GetDisplayRange(Im, Mask, RoiNr, dispMode, &minDisp, &maxDisp);

        Mat ImF;
        Im.convertTo(ImF, CV_64F);
        ImToShow = ShowImageF64PseudoColor(ImF, minDisp, maxDisp);
        ui->textEditOut->append("range " + QString::number(minDisp) + " - " + QString::number(maxDisp));

    }
    else
        ImToShow = Im.clone();

    if (dispScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), dispScale, dispScale, INTER_AREA);
    imwrite(FileName, ImToShow);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::TiffRoiFromRed()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return;
    }

    if(ImIn.depth() != CV_8U)
    {
        ui->textEditOut->append("Improper image type");
        return;
    }
    if(ImIn.channels() != 3)
    {
        ui->textEditOut->append("Iproper number of channels");
        return;
    }

    ImOut.release();
    ImOut = Mat::zeros(ImIn.size(), CV_16U);
    int maxXY = ImIn.cols * ImIn.rows;

    unsigned short *wImOut = (unsigned short *)ImOut.data;
    unsigned char *wImIn = (unsigned char *)ImIn.data;
    for (int i = 0; i < maxXY; i ++)
    {
        char B = *wImIn;
        wImIn++;
        char G = *wImIn;
        wImIn++;
        char R = *wImIn;
        wImIn++;

        if (B != G)
            *wImOut = 1;
        wImOut++;
    }
    if(ui->checkBoxShowOutput->checkState())
        ShowsScaledImage(ShowRegion(ImOut), "Output Image",displayScale);
    if(ui->checkBoxSaveOutput->checkState())
    {
        path fileToSave = OutFolder;
        path fileToOpen = FileName;
        fileToSave.append(fileToOpen.stem().string());
        imwrite(fileToSave.string() + ".tif" ,ImOut);

    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ImageResize()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return;
    }
    ImOut.release();
    cv::resize(ImIn,ImOut,Size(), resizeScale, resizeScale, resizeInterpolation);
    ui->textEditOut->append(QString::fromStdString(InterpolationToString(resizeInterpolation)));

    if(ui->checkBoxShowOutMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImOut)));
    if(ui->checkBoxShowOutput->checkState())
        ShowsScaledImage(ImOut, "Output Image", displayScale,ui->comboBoxDisplayRange->currentIndex());

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ImageLinearOperation()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return;
    }
    ImOut.release();
    Mat ImIn32S;

    if(ui->checkBoxPlainImage->checkState())
        ImIn32S = Mat::ones(ImIn.size(), CV_32S)*(int32_t)(ui->doubleSpinBoxIntensityScale->value());
    else
        ImIn.convertTo(ImIn32S,CV_32S,ui->doubleSpinBoxIntensityScale->value(),0);

    if(ui->checkBoxShowHist->checkState())
    {
        HistogramInteger ImInHist;

        ImInHist.FromMat32S(ImIn32S);
        Mat HistPlot = ImInHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                          ui->spinBoxHistScaleCoef->value(),
                                          ui->spinBoxHistBarWidth->value());
        imshow("Intensity histogram Input",HistPlot);
        ImInHist.Release();
    }

    ImOut = ImIn32S;
    Mat ImNoise;

    if(ui->checkBoxAddNoise->checkState())
    {

        double noiseStd = ui->doubleSpinBoxGaussNianoiseSigma->value();
        ImNoise = Mat::zeros(ImIn.size(), CV_32S);
        int32_t *wImNoise = (int32_t *)ImNoise.data;
        int maxX = ImNoise.cols;
        int maxY = ImNoise.rows;
        int maxXY = maxX * maxY;
        for(int i = 0; i < maxXY; i++)
        {

                *wImNoise = (int32_t)round(RandomGenNormDistribution->operator()() * noiseStd);
                wImNoise ++;
        }



        if(ui->checkBoxShowHist->checkState())
        {
            HistogramInteger IntensityHist;

            IntensityHist.FromMat32S(ImNoise);
            Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                              ui->spinBoxHistScaleCoef->value(),
                                              ui->spinBoxHistBarWidth->value());
            imshow("Intensity histogram Noise",HistPlot);

            IntensityHist.Release();
        }

        ImOut += ImNoise;
        ImNoise.release();
    }
    if(ui->checkBoxAddUniformNoise->checkState())
    {
        ImNoise = Mat::zeros(ImIn.size(), CV_32S);

        int32_t *wImNoise = (int32_t *)ImNoise.data;
        int maxX = ImNoise.cols;
        int maxY = ImNoise.rows;
        int maxXY = maxX * maxY;
        for(int i = 0; i < maxXY; i++)
        {
                *wImNoise = RandomGenUniformDistribution->operator()();
                wImNoise ++;

        }

        if(ui->checkBoxShowHist->checkState())
        {
            HistogramInteger IntensityHist;

            IntensityHist.FromMat32S(ImNoise);
            Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                              ui->spinBoxHistScaleCoef->value(),
                                              ui->spinBoxHistBarWidth->value());
            imshow("Intensity histogram Noise",HistPlot);

            IntensityHist.Release();
        }
        ImOut += ImNoise;
        ImNoise.release();
    }


    if(ui->checkBoxAddRician->checkState())
    {

        double ricianS = ui->doubleSpinBoxRicianS->value();
        Mat ImTemp;
        ImOut.copyTo(ImTemp);

        int32_t *wImOut = (int32_t *)ImOut.data;
        int maxX = ImOut.cols;
        int maxY = ImOut.rows;
        for(int y = 0; y < maxY; y++)
        {
            for(int x = 0; x < maxX; x++)
            {
                double valIm = (double)*wImOut;
                double valRNG1 =  RandomGenNormDistribution->operator()() * ricianS  + valIm;
                double valRNG2 =  RandomGenNormDistribution->operator()() * ricianS ;
                double valOut = round(sqrt(valRNG1 * valRNG1 + valRNG2 * valRNG2));
                *wImOut = valOut;
                wImOut ++;
            }
        }

        Mat ImNoise = ImOut - ImTemp;
        ImTemp.release();
        if(ui->checkBoxShowHist->checkState())
        {
            HistogramInteger IntensityHist;

            IntensityHist.FromMat32S(ImNoise);
            Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                              ui->spinBoxHistScaleCoef->value(),
                                              ui->spinBoxHistBarWidth->value());
            imshow("Intensity histogram Noise",HistPlot);

            IntensityHist.Release();
        }
        ImNoise.release();
    }

    if(ui->checkBoxAddGradient->checkState())
    {
        int32_t *wImOut = (int32_t *)ImOut.data;
        int maxX = ImOut.cols;
        int maxY = ImOut.rows;
        for(int y = 0; y < maxY; y++)
        {
            for(int x = 0; x < maxX; x++)
            {
                double val;
                switch(ui->comboBoxGradientDirection->currentIndex())
                {
                case 1:
                    val = y * ui->doubleSpinBoxGradNominator->value()/ui->doubleSpinBoxGradDenominator->value();
                    break;
                case 2:
                    val = (x + y) * ui->doubleSpinBoxGradNominator->value()/ui->doubleSpinBoxGradDenominator->value();
                    break;
                default:
                    val = x * ui->doubleSpinBoxGradNominator->value()/ui->doubleSpinBoxGradDenominator->value();
                    break;
                }

                if(val < 0.0)
                    val = 0.0;
                if(val > 65535.0)
                    val = 65535.0;
                *wImOut += (int32_t)val;
                wImOut ++;
            }
        }
    }

    ImOut = ImOut + (int32_t)round(ui->doubleSpinBoxIntOffset->value());


    ImOut.convertTo(ImOut,CV_16U,1.0,0.0);

    if(ui->checkBoxShowOutput->checkState())
        ShowsScaledImage(ImOut, "Output Image", displayScale,ui->comboBoxDisplayRange->currentIndex());

    if(ui->checkBoxShowHist->checkState())
    {
        HistogramInteger IntensityHist;

        IntensityHist.FromMat16U(ImOut);
        Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                          ui->spinBoxHistScaleCoef->value(),
                                          ui->spinBoxHistBarWidth->value());
        //ui->textEditOut->append(QString::fromStdString(IntensityHist.GerString()));
        imshow("Intensity histogram Output",HistPlot);

        IntensityHist.Release();
    }
    if(ui->checkBoxSaveOutput->checkState())
    {
        path fileToOpen(FileName);
        string OutFileName = fileToOpen.stem().string();
        path fileToSave = OutFolder;
        if(ui->checkBoxAddNoise->checkState())
        {
            OutFileName += "GN";
            OutFileName += ui->doubleSpinBoxGaussNianoiseSigma->text().toStdString();
        }
        if(ui->checkBoxAddRician->checkState())
        {
            OutFileName += "RN";
            OutFileName += ui->doubleSpinBoxRicianS->text().toStdString();
        }
        if(ui->checkBoxAddUniformNoise->checkState())
        {
            OutFileName += "UN";
            OutFileName += to_string(ui->spinBoxUniformNoiseStart->value());
            OutFileName += "-";
            OutFileName += to_string(ui->spinBoxUniformNoiseStop->value());
        }
        if(ui->checkBoxAddGradient->checkState())
        {
            OutFileName += "Gr";
            //OutFileName += to_string(ui->spinBoxGradientNominator->value());
            //OutFileName += "over";
            //OutFileName += to_string(ui->spinBoxGradientDenominator->value());
        }

        OutFileName += ".tiff";
        fileToSave.append(OutFileName);

        imwrite(fileToSave.string(),ImOut);
    }


}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::CreateROI()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return;
    }
    Mat Mask = Mat::zeros(ImIn.size(),CV_16U);

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;

    int maxXY = maxX * maxY;

    int roiOffsetX = ui->spinBoxRoiOffset->value();
    int roiOffsetY = ui->spinBoxRoiOffset->value();

    int roiSize = ui->spinBoxRoiSize->value();
    int roiShift = ui->spinBoxRoiShift->value();


    int firstRoiY = roiOffsetY;
    int lastRoiY = maxY - roiSize / 2;
    int firstRoiX = roiOffsetX;
    int lastRoiX = maxX - roiSize / 2;


    //int roiNrSel =
    int roiNr = 1;
    int skip = 0;

    for (int y = firstRoiY; y < lastRoiY; y += roiShift)
    {
        for (int x = firstRoiX; x < lastRoiX; x += roiShift)
        {
            if(ui->checkBoxReducedROI->checkState() && !ui->checkBoxReducedROIComplement->checkState())
            {
                if (skip <= 0)
                    skip = ui->spinBoxSkipCount->value();
                else
                {
                    skip--;
                    continue;
                }
            }
            if(ui->checkBoxReducedROI->checkState() && ui->checkBoxReducedROIComplement->checkState())
            {
                if (skip <= 0)
                {
                    skip = ui->spinBoxSkipCount->value();
                    continue;
                }
                else
                {
                    skip--;
                }
            }
            switch (ui->comboBoxRoiShape->currentIndex())
            {
            case 1:
                circle(Mask,Point(x,y),roiSize/2,roiNr,-1);
                break;
            default:
                int roiLeftTopBorderOffset = roiSize / 2 ;
                int roiRigthBottomBorderOffset =  roiSize - roiSize / 2 - 1 ;
                rectangle(Mask, Point(x - roiLeftTopBorderOffset, y - roiLeftTopBorderOffset),
                    Point(x + roiRigthBottomBorderOffset, y + roiRigthBottomBorderOffset),
                    roiNr,-1);

                break;
            }
            roiNr++;
        }
    }



    int *ROISizes = new int[65535];
    for(int i = 0; i < 65535; i++)
    {
        ROISizes[i] = 0;
    }

    uint16_t maxRoiNr = 0;
    uint16_t *wMask = (uint16_t *)Mask.data;
    for(int i = 0; i < maxXY; i++)
    {
        uint16_t roiNr = *wMask;
        ROISizes[roiNr]++;
        if(maxRoiNr < roiNr)
        {
            maxRoiNr = roiNr;
        }
        wMask++;
    }

    ui->textEditOut->append("Max ROI Nr "+QString::number(maxRoiNr));

    if(ui->checkBoxShowOutput->checkState())
    {
        ShowsScaledImage(ShowRegion(Mask), "Output Image",displayScale);
    }


    if(ui->checkBoxSaveROIbmp->checkState())
    {
        path fileToSave = OutFolder;
        string RoiName = "ROI_";
        switch(ui->comboBoxRoiShape->currentIndex())
        {
        case 1:
            RoiName += "Cir";
            break;
        default:
            RoiName += "Rct";
            break;
        }

        RoiName += to_string(ui->spinBoxRoiSize->value());
        RoiName += "Cnt";
        RoiName += to_string(maxRoiNr);
        RoiName +=  ".bmp";
        fileToSave.append(RoiName);
        imwrite(fileToSave.string(),ShowRegion(Mask));
        //SaveScaledImage(ShowRegion(Mask), fileToSave.string(), ui->doubleSpinBoxROIScale->value(), ui->comboBoxDisplayRange->currentIndex());
    }


    ui->spinBoxRoiNr->setMaximum(maxRoiNr);
    if(ui->checkBoxShowHist->checkState()|| ui->checkBoxSaveRoiHistogram->checkState())
    {
        ImIn.convertTo(ImOut,CV_16U);
        HistogramInteger IntensityHist;

        if(ui->checkBoxFixtRangeHistogram->checkState())
            IntensityHist.FromMat16ULimit(ImOut, Mask, ui->spinBoxRoiNr->value(),
                                          ui->spinBoxMinHist->value(),
                                          ui->spinBoxMaxHist->value());
        else
            IntensityHist.FromMat16U(ImOut,Mask,ui->spinBoxRoiNr->value());

        if(ui->checkBoxShowHist->checkState())
        {
            Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                              ui->spinBoxHistScaleCoef->value(),
                                              ui->spinBoxHistBarWidth->value());
            //ui->textEditOut->append(QString::fromStdString(IntensityHist.GerString()));
            imshow("Intensity histogram Output",HistPlot);
        }


        if(ui->checkBoxSaveRoiHistogram->checkState())
        {
            path fileToOpen(FileName);
            string RoiImName = fileToOpen.stem().string();

            path fileToSave = OutFolder;
            switch(ui->comboBoxRoiShape->currentIndex())
            {
            case 1:
                RoiImName += "Cir";
                break;
            default:
                RoiImName += "Rct";
                break;
            }

            RoiImName += to_string(ui->spinBoxRoiSize->value());
            RoiImName += "Cnt";
            RoiImName += to_string(maxRoiNr);
            RoiImName += "Nr";
            RoiImName += to_string(ui->spinBoxRoiNr->value());
            RoiImName +=  ".txt";
            fileToSave.append(RoiImName);

            std::ofstream out (fileToSave.string());
            out << IntensityHist.GetString();
            out.close();

        }


        if(ui->checkBoxSaveStatistics->checkState())
        {
            OutStringStat.clear();
            OutStringStat = IntensityHist.StatisticStringOut();
            ui->textEditOut->append(QString::fromStdString(OutStringStat));
        }
        IntensityHist.Release();
    }

    if(ui->checkBoxShowNormalisedROI->checkState() || ui->checkBoxSaveNormalisedRoiImage->checkState()||ui->checkBoxShowBinedROI->checkState())
    {
        uint16_t roiNr = (uint16_t)ui->spinBoxRoiNr->value();
        int roiMaxX = 0;
        int roiMinX = maxX;
        int roiMaxY = 0;
        int roiMinY = maxY;
        wMask = (uint16_t *)Mask.data;
        for (int y = 0; y < maxY; y++)
        {
            for (int x = 0; x < maxX; x++)
            {
                if(*wMask == roiNr)
                {
                    if(roiMaxX < x)
                        roiMaxX = x;
                    if(roiMinX > x)
                        roiMinX = x;
                    if(roiMaxY < y)
                        roiMaxY = y;
                    if(roiMinY > y)
                        roiMinY = y;
                }
                wMask++;
            }
        }
        Mat SmallIm,SmallMask;
        ImIn(Rect(roiMinX,roiMinY, roiMaxX-roiMinX+1, roiMaxY-roiMinY+1)).copyTo(SmallIm);
        Mask(Rect(roiMinX,roiMinY, roiMaxX-roiMinX+1, roiMaxY-roiMinY+1)).copyTo(SmallMask);

        if(ui->checkBoxShowNormalisedROI->checkState())
            ShowsScaledImage(SmallIm, SmallMask, "ROI small", ui->doubleSpinBoxROIScale->value(), roiNr, ui->comboBoxDisplayRange->currentIndex() );
                         //(Mat Im, Mat Mask, string ImWindowName, double dispScale, uint16_t RoiNr, int dispMode )

        if(ui->checkBoxShowBinedROI->checkState()||ui->checkBoxSaveBinnedROIImage->checkState())
        {

            Mat ImToShow;

            double minNorm = 0.0;
            double maxNorm = 255.0;

            switch(ui->comboBoxROINorm->currentIndex())
            {


            case 1:
                NormParamsMeanP3Std(SmallIm, SmallMask, (uint16_t)ui->spinBoxRoiNr->value(), &maxNorm, &minNorm);
                break;
            case 2:
                NormParams1to99perc(SmallIm, SmallMask,(uint16_t)ui->spinBoxRoiNr->value(), &maxNorm, &minNorm);
                break;
            default:
                NormParamsMinMax(SmallIm, SmallMask, (uint16_t)ui->spinBoxRoiNr->value(), &maxNorm, &minNorm);
                break;
            }
            int binCount = (int)pow(2,ui->spinBoxROIBitPerPix->value());

            Mat ImBinned = CreateNormalisedImage16U(SmallIm,minNorm,maxNorm,binCount);

            ImToShow = ShowImage16PseudoColor(ImBinned,0.0,binCount-1);

            if (ui->doubleSpinBoxROIScale->value() != 1.0)
                cv::resize(ImToShow,ImToShow,Size(), ui->doubleSpinBoxROIScale->value(), ui->doubleSpinBoxROIScale->value(), INTER_AREA);

            imshow("Im Binned", ImToShow);



            if(ui->checkBoxSaveBinnedROIImage->checkState())
            {
                path fileToOpen(FileName);
                string RoiImName = fileToOpen.stem().string();

                path fileToSave = OutFolder;
                switch(ui->comboBoxRoiShape->currentIndex())
                {
                case 1:
                    RoiImName += "Cir";
                    break;
                default:
                    RoiImName += "Rct";
                    break;
                }

                RoiImName += to_string(ui->spinBoxRoiSize->value());
                RoiImName += "Cnt";
                RoiImName += to_string(maxRoiNr);
                RoiImName += "Nr";
                RoiImName += to_string(ui->spinBoxRoiNr->value());
                switch(ui->comboBoxROINorm->currentIndex())
                {
                case 1:
                    RoiImName += "NormMeanPM3STD";
                    break;
                case 2:
                    RoiImName += "Norm1_99Perc";
                    break;
                default:
                    RoiImName += "NormMinMax";
                    break;
                }
                RoiImName += "BpP";
                RoiImName += to_string(ui->spinBoxROIBitPerPix->value());
                RoiImName +=  ".bmp";
                fileToSave.append(RoiImName);
                imwrite(fileToSave.string(),ImToShow);
            }

            if(ui->checkBoxShowHist->checkState() || ui->checkBoxSaveBinnedROIHist->checkState())
            {
                HistogramInteger IntensityHist;

                IntensityHist.FromMat16ULimit(ImBinned,SmallMask,ui->spinBoxRoiNr->value(),0,binCount+1);

                if(ui->checkBoxShowHist->checkState())
                {
                    Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                                      ui->spinBoxHistScaleCoef->value(),
                                                      ui->spinBoxHistBarWidth->value());
                    imshow("Intensity histogram ROI Binned",HistPlot);
                }


                if(ui->checkBoxSaveBinnedROIHist->checkState())
                {
                    path fileToOpen(FileName);
                    string RoiImName = fileToOpen.stem().string();

                    path fileToSave = OutFolder;
                    switch(ui->comboBoxRoiShape->currentIndex())
                    {
                    case 1:
                        RoiImName += "Cir";
                        break;
                    default:
                        RoiImName += "Rct";
                        break;
                    }

                    RoiImName += to_string(ui->spinBoxRoiSize->value());
                    RoiImName += "Cnt";
                    RoiImName += to_string(maxRoiNr);
                    RoiImName += "Nr";
                    RoiImName += to_string(ui->spinBoxRoiNr->value());
                    switch(ui->comboBoxROINorm->currentIndex())
                    {
                    case 1:
                        RoiImName += "NormMeanPM3STD";
                        break;
                    case 2:
                        RoiImName += "Norm1_99Perc";
                        break;
                    default:
                        RoiImName += "NormMinMax";
                        break;
                    }
                    RoiImName += "BpP";
                    RoiImName += to_string(ui->spinBoxROIBitPerPix->value());
                    RoiImName +=  ".txt";
                    fileToSave.append(RoiImName);

                    std::ofstream out (fileToSave.string());
                    out << IntensityHist.GetString();
                    out.close();

                }
                IntensityHist.Release();
            }

        }

        if(ui->checkBoxSaveNormalisedRoiImage->checkState())
        {
            path fileToOpen(FileName);
            string RoiImName = fileToOpen.stem().string();

            path fileToSave = OutFolder;
            switch(ui->comboBoxRoiShape->currentIndex())
            {
            case 1:
                RoiImName += "Cir";
                break;
            default:
                RoiImName += "Rct";
                break;
            }

            RoiImName += to_string(ui->spinBoxRoiSize->value());
            RoiImName += "Cnt";
            RoiImName += to_string(maxRoiNr);


            RoiImName += "Nr";
            RoiImName += to_string(ui->spinBoxRoiNr->value());
            switch(ui->comboBoxDisplayRange->currentIndex())
            {
            case 1:
                RoiImName += "NormFixed";
                break;
            case 2:
                RoiImName += "NormMeanPM3STD";
                break;
            case 3:
                RoiImName += "Norm1_99Perc";
                break;
            case 4:
                RoiImName += "NormMinMax";
                break;
            default:
                RoiImName += "NormNone";
                break;
            }
            RoiImName += "BpP";
            RoiImName += to_string(ui->spinBoxROIBitPerPix->value());

            RoiImName +=  ".bmp";
            fileToSave.append(RoiImName);
            SaveScaledImage(SmallIm, SmallMask, fileToSave.string(), ui->doubleSpinBoxROIScale->value(), roiNr, ui->comboBoxDisplayRange->currentIndex());
        }
    }



    path fileToOpen(FileName);
    string RoiName = fileToOpen.stem().string();

    if(ui->checkBoxSaveRoi->checkState())
    {
        vector <MR2DType*> ROIVect;
        int begin[MR2DType::Dimensions];
        int end[MR2DType::Dimensions];
        begin[0] = 0;
        begin[1] = 0;
        end[0] = maxX-1;
        end[1] = maxY-1;

        MR2DType *ROI;

        for(int roiNr = 1; roiNr <=maxRoiNr; roiNr++)
        {
            ROI = new MR2DType(begin, end);

            MazdaRoiIterator<MR2DType> iteratorKL(ROI);
            wMask = (uint16_t *)Mask.data;
            while(! iteratorKL.IsBehind())
            {
                if (*wMask == roiNr)
                    iteratorKL.SetPixel();
                ++iteratorKL;
                wMask++;
            }

            ROI->SetName(RoiName);
            ROI->SetColor(RegColorsRGB[(roiNr-1)%16]);

            ROIVect.push_back(ROI);
        }

        path fileToSave = OutFolder;
        switch(ui->comboBoxRoiShape->currentIndex())
        {
        case 1:
            RoiName += "Cir";
            break;
        default:
            RoiName += "Rct";
            break;
        }

        RoiName += to_string(ui->spinBoxRoiSize->value());
        RoiName += "Cnt";
        RoiName += to_string(maxRoiNr);
        RoiName +=  ".roi";
        fileToSave.append(RoiName);

        MazdaRoiIO<MR2DType>::Write(fileToSave.string(), &ROIVect, NULL);
        while(ROIVect.size() > 0)
        {
             delete ROIVect.back();
             ROIVect.pop_back();
        }
    }

}

//------------------------------------------------------------------------------------------------------------------------------
string MainWindow::CreateMaZdaScript()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return "";
    }
    Mat Mask ;

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;

    path ROIFile = ImageFolder;
    path ImageFileName(FileName);
    ROIFile.append("/" + ui->lineEditMaZdaROIFolder->text().toStdString() + ImageFileName.stem().string() + ".roi");

   // ROIFile
    if(exists(ROIFile))
    {
        Mask =  LoadROI(ROIFile, maxX, maxY);

        ui->textEditOut->append("Valid Roi");
    }
    else
    {
        ui->textEditOut->append("No Roi For The Frame");

    }

    if(ui->checkBoxShowOutput->checkState())
    {
        double minDisp = 0.0;
        double maxDisp = 255.0;

        switch(ui->comboBoxDisplayRange->currentIndex())
        {
        case 1:
            minDisp = ui->doubleSpinBoxFixMinDisp->value();
            maxDisp = ui->doubleSpinBoxFixMaxDisp->value();

            break;
        case 2:
            NormParamsMinMax(ImIn, &maxDisp, &minDisp);
            break;
        case 3:
            NormParamsMeanP3Std(ImIn, &maxDisp, &minDisp);
            break;
        case 4:
            NormParams1to99perc(ImIn, &maxDisp, &minDisp);
            break;
        default:
            break;
        }
        Mat ImShowGray = ShowImage16Gray(ImIn,minDisp,maxDisp);
        Mat ImShow = ShowSolidRegionOnImage(GetContour5(Mask),ImShowGray);
        ShowsScaledImage(ImShow, "Output Image",displayScale);
    }
    if(ui->checkBoxShowHist->checkState())
    {
        ImIn.convertTo(ImOut,CV_16U);
        HistogramInteger IntensityHist;

        IntensityHist.FromMat16U(ImOut,Mask,2);
        Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                          ui->spinBoxHistScaleCoef->value(),
                                          ui->spinBoxHistBarWidth->value());
        //ui->textEditOut->append(QString::fromStdString(IntensityHist.GerString()));
        imshow("Intensity histogram Output",HistPlot);

        IntensityHist.Release();
    }

    string out = ui->lineEditMaZdaFileLocation->text().toStdString();
    out += " -m roi -i ";
    out += ui->lineEditMaZdaInFilesFolder->text().toStdString();
    out += ImageFileName.filename().string();
    out += " -r ";
    //out += ui->lineEditMaZdaInFilesFolder->text().toStdString();
    out += ui->lineEditMaZdaROIFolder->text().toStdString();
    out += ImageFileName.stem().string();
    out += ".roi";
    if (ui->listWidgetImageFiles->currentRow())
    {
        out += " -a ";
    }
    out += " -o ";
    out += ui->lineEditMaZdaOutFileName->text().toStdString();
    out += ui->lineEditMaZdaOptionsFile->text().toStdString();
    out += ".cvs";

    if (ui->listWidgetImageFiles->currentRow()==0)
    {
        out += " -f ";
        out += ui->lineEditMaZdaOptionsDir->text().toStdString();
        out += ui->lineEditMaZdaOptionsFile->text().toStdString();
        out += ".";
        out += ui->lineEditMaZdaOptionsExtension->text().toStdString();
    }
    out += "\n";
    return out;
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ViewRoi()
{
    if(ImIn.empty())
    {
        ui->textEditOut->append("Empty Image");
        return ;
    }
    Mat Mask ;

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;

    path ROIFile = ImageFolder;
    path ImageFileName(FileName);
    ROIFile.append("/" + ui->lineEditViewROIFolder->text().toStdString() + ImageFileName.stem().string() + ".roi");

    if(exists(ROIFile))
    {
        Mask =  LoadROI(ROIFile, maxX, maxY);
        ui->textEditOut->append("Valid Roi");
    }
    else
    {
        ui->textEditOut->append("No Roi For The Frame");
        return;
    }

    if(ui->checkBoxShowOutput->checkState())
    {

        double minDisp, maxDisp;
        GetDisplayRange(ImIn, ui->comboBoxDisplayRange->currentIndex(), &minDisp, &maxDisp);

        Mat ImShowGray = ShowImage16Gray(ImIn,minDisp,maxDisp);
        Mat ImShow;
        if(ui->checkBoxShowROIOnImage->checkState())
            ImShow = ShowSolidRegionOnImage(GetContour5(Mask),ImShowGray);
        else
            ImShow = ImShowGray;
        ShowsScaledImage(ImShow, "Output Image",displayScale);
        if(ui->checkBoxViewSaveBinnedROIImage->checkState())
        {
            path fileToOpen(FileName);
            string RoiImName = fileToOpen.stem().string();

            path fileToSave = OutFolder;
            //RoiImName += to_string(maxRoiNr);
            RoiImName += "Nr";
            RoiImName += to_string(ui->spinBoxViewROINr->value());
            RoiImName += "NormNone";

            RoiImName +=  ".bmp";

            fileToSave.append(RoiImName);
            SaveScaledImage(ImShow, fileToSave.string(),displayScale,0);


        }
    }

    if(ui->checkBoxShowHist->checkState())
    {
        Mat ImTemp;
        ImIn.convertTo(ImTemp,CV_16U);
        HistogramInteger IntensityHist;
        if(ui->checkBoxFixtRangeHistogram->checkState())
            IntensityHist.FromMat16ULimit(ImTemp,Mask,1, ui->spinBoxMinHist->value(),ui->spinBoxMaxHist->value());
        else
            IntensityHist.FromMat16U(ImTemp,Mask,1);

        Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                          ui->spinBoxHistScaleCoef->value(),
                                          ui->spinBoxHistBarWidth->value());
        imshow("Intensity histogram Input",HistPlot);


        if(ui->checkBoxVewSaveRoiBinnedHistogram->checkState())
        {
            path fileToOpen(FileName);
            string RoiImName = fileToOpen.stem().string();

            path fileToSave = OutFolder;
            //RoiImName += to_string(maxRoiNr);
            RoiImName += "Nr";
            RoiImName += to_string(ui->spinBoxViewROINr->value());
            RoiImName += "NormNone";

            RoiImName +=  ".txt";

            fileToSave.append(RoiImName);

            std::ofstream out (fileToSave.string());
            out << IntensityHist.GetString();
            out.close();
        }
        IntensityHist.Release();
    }

    if(ui->checkBoxViewRoiShowBined->checkState())
    {
        Mat ImToShow;

        double minNorm = 0.0;
        double maxNorm = 255.0;

        switch(ui->comboBoxViewROINorm->currentIndex())
        {
        case 1:
            NormParamsMeanP3Std(ImIn, Mask, (uint16_t)ui->spinBoxViewROINr->value(), &maxNorm, &minNorm);
            break;
        case 2:
            NormParams1to99perc(ImIn, Mask,(uint16_t)ui->spinBoxViewROINr->value(), &maxNorm, &minNorm);
            break;
        default:
            NormParamsMinMax(ImIn, Mask, (uint16_t)ui->spinBoxViewROINr->value(), &maxNorm, &minNorm);
            break;
        }
        int binCount = (int)pow(2,ui->spinBoxViewROIBitPerPixel->value());

        Mat ImBinned = CreateNormalisedImage16U(ImIn,minNorm,maxNorm,binCount);

        ImToShow = ShowImage16PseudoColor(ImBinned,0.0,binCount-1);

        if (displayScale != 1.0)
            cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);

        imshow("Im Binned", ImToShow);



        if(ui->checkBoxViewSaveBinnedROIImage->checkState())
        {
            path fileToOpen(FileName);
            string RoiImName = fileToOpen.stem().string();


            //RoiImName += to_string(maxRoiNr);
            RoiImName += "Nr";
            RoiImName += to_string(ui->spinBoxViewROINr->value());
            switch(ui->comboBoxViewROINorm->currentIndex())
            {
            case 1:
                RoiImName += "NormMeanPM3STD";
                break;
            case 2:
                RoiImName += "Norm1_99Perc";
                break;
            default:
                RoiImName += "NormMinMax";
                break;
            }
            RoiImName += "BpP";
            RoiImName += to_string(ui->spinBoxViewROIBitPerPixel->value());
            RoiImName +=  ".bmp";
            path fileToSave(OutFolder);
            fileToSave.append(RoiImName);
            imwrite(fileToSave.string(),ImToShow);
        }

        if(ui->checkBoxShowHist->checkState() || ui->checkBoxVewSaveRoiBinnedHistogram->checkState())
        {
            HistogramInteger IntensityHist;

            IntensityHist.FromMat16ULimit(ImBinned, Mask, ui->spinBoxViewROINr->value(),0 , binCount-1);

            if(ui->checkBoxShowHist->checkState())
            {
                Mat HistPlot = IntensityHist.Plot(ui->spinBoxHistScaleHeight->value(),
                                                  ui->spinBoxHistScaleCoef->value(),
                                                  ui->spinBoxHistBarWidth->value());
                imshow("Intensity histogram ROI Binned",HistPlot);
            }


            if(ui->checkBoxVewSaveRoiBinnedHistogram->checkState())
            {
                path fileToOpen(FileName);
                string RoiImName = fileToOpen.stem().string();

                path fileToSave = OutFolder;
                //RoiImName += to_string(maxRoiNr);
                RoiImName += "Nr";
                RoiImName += to_string(ui->spinBoxViewROINr->value());
                switch(ui->comboBoxViewROINorm->currentIndex())
                {
                case 1:
                    RoiImName += "NormMeanPM3STD";
                    break;
                case 2:
                    RoiImName += "Norm1_99Perc";
                    break;
                default:
                    RoiImName += "NormMinMax";
                    break;
                }
                RoiImName += "BpP";
                RoiImName += to_string(ui->spinBoxViewROIBitPerPixel->value());
                RoiImName +=  ".txt";

                fileToSave.append(RoiImName);

                std::ofstream out (fileToSave.string());
                out << IntensityHist.GetString();
                out.close();
            }
            IntensityHist.Release();
        }
    }
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          Slots
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::on_pushButtonOpenImageFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(QString::fromStdString(ImageFolder.string()));

    if(dialog.exec())
    {
        ImageFolder = dialog.directory().path().toStdWString();
    }
    else
        return;
    OpenImageFolder();
}

void MainWindow::on_pushButtonOpenOutFolder_clicked()
{
    QFileDialog dialogOF(this, "Open Folder");
    dialogOF.setFileMode(QFileDialog::Directory);
    dialogOF.setDirectory(QString::fromStdString(OutFolder.string()));

    if(dialogOF.exec())
    {
        OutFolder = dialogOF.directory().path().toStdWString();
    }
    else
        return;

    if (!exists(OutFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + OutFolder.string()+ " not exists "));
        OutFolder = "d:\\";
    }
    if (!is_directory(OutFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + OutFolder.string()+ " This is not a directory path "));
        OutFolder = "C:\\Data\\";
    }
    ui->lineEditOutFolder->setText(QString::fromStdString(OutFolder.string()));
}

void MainWindow::on_lineEditRegexImageFile_returnPressed()
{
    OpenImageFolder();
}

void MainWindow::on_listWidgetImageFiles_currentTextChanged(const QString &currentText)
{
    path fileToOpen = ImageFolder;
    fileToOpen.append(currentText.toStdString());
    FileName = fileToOpen.string();
    ModeSelect();

}

void MainWindow::on_spinBoxScaleBase_valueChanged(int arg1)
{
    displayScale = pow(double(arg1), double(ui->spinBoxScalePower->value()));
    ModeSelect();
}

void MainWindow::on_spinBoxScalePower_valueChanged(int arg1)
{
    displayScale = pow(double(ui->spinBoxScaleBase->value()), double(arg1));
    ModeSelect();
}

void MainWindow::on_tabWidgetMode_currentChanged(int index)
{
    operationMode = index;
    ModeSelect();
}


void MainWindow::on_lineEditImageScale_returnPressed()
{
    if(ui->checkBoxKeeprequestedPixelSize->checkState())
        return;
    bool properConversion;
    double newScale = ui->lineEditImageScale->text().toDouble(&properConversion);
    if(!properConversion)
    {
        ui->lineEditImageScale->setText("iproper value enter again");
        return;
    }
    if(newScale < 0.01)
    {
        ui->lineEditImageScale->setText("to small scale");
        return;
    }
    if(newScale > 20.0)
    {
        ui->lineEditImageScale->setText("to large scale");
        return;
    }
    resizeScale = newScale;
    ModeSelect();
}

void MainWindow::on_comboBoxImageInterpolationMethod_currentIndexChanged(int index)
{
    resizeInterpolation = index;
    ModeSelect();
}

void MainWindow::on_checkBoxShowOutput_toggled(bool checked)
{
   ModeSelect();
}

void MainWindow::on_pushButtonSaveResized_clicked()
{
    if (!exists(OutFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + OutFolder.string()+ " not exists "));
        return;
    }
    if (!is_directory(OutFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + OutFolder.string()+ " This is not a directory path "));
        return;
    }

    path fileToSave = OutFolder;
    path fileToOpen(FileName);
    fileToSave.append(fileToOpen.stem().string()+ "resizedScale" + to_string(resizeScale) + ".tif");

    imwrite(fileToSave.string(),ImOut);




}

void MainWindow::on_lineEditPixelSize_returnPressed()
{

    if(!ui->checkBoxKeeprequestedPixelSize->checkState())
        return;
    bool properConversion;
    double newPixelSize = ui->lineEditImageScale->text().toDouble(&properConversion);
    if(!properConversion)
    {
        ui->lineEditPixelSize->setText("iproper value enter again");
        return;
    }

    ModeSelect();

}

void MainWindow::on_checkBoxLoadAnydepth_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_checkBoxSaveOutput_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_comboBoxDisplayRange_currentIndexChanged(int index)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxIntOffset_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxFixMinDisp_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxFixMaxDisp_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxShowHist_stateChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxHistBarWidth_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxHistScaleHeight_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxHistScaleCoef_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxAddNoise_stateChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxGaussNianoiseSigma_valueChanged(double arg1)
{
    /*
    delete rngNormalDist;
    delete normalDistribution;
    delete RandomGenNormDistribution;

    rngNormalDist = new boost::minstd_rand(time(0));
    normalDistribution = new boost::normal_distribution<>(0.0, ui->doubleSpinBoxGaussNianoiseSigma->value());
    RandomGenNormDistribution = new boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<>>(*rngNormalDist, *normalDistribution);
    */
    ModeSelect();
}

void MainWindow::on_checkBoxAddNoise_toggled(bool checked)
{
    /*
    ready = 0;
    ui->checkBoxAddRician->setCheckState(Qt::Unchecked);
    ui->checkBoxAddUniformNoise->setCheckState(Qt::Unchecked);
    ready = 1;
    if(ui->checkBoxAddUniformNoise->checkState())
        ui->checkBoxAddUniformNoise->setCheckState(Qt::Checked);
    */
    ModeSelect();
}

void MainWindow::on_checkBoxAddGradient_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_comboBoxGradientDirection_currentIndexChanged(int index)
{
    ModeSelect();
}

void MainWindow::on_spinBoxUniformNoiseStart_valueChanged(int arg1)
{
    delete rngUniformDist;
    delete uniformDistribution;
    delete RandomGenUniformDistribution;

    rngUniformDist = new boost::minstd_rand(time(0));
    uniformDistribution = new boost::uniform_int<>(ui->spinBoxUniformNoiseStart->value(), ui->spinBoxUniformNoiseStop->value());
    RandomGenUniformDistribution = new boost::variate_generator<boost::minstd_rand&, boost::uniform_int<>>(*rngUniformDist, *uniformDistribution);

    ModeSelect();
}

void MainWindow::on_spinBoxUniformNoiseStop_valueChanged(int arg1)
{
    delete rngUniformDist;
    delete uniformDistribution;
    delete RandomGenUniformDistribution;

    rngUniformDist = new boost::minstd_rand(time(0));
    uniformDistribution = new boost::uniform_int<>(ui->spinBoxUniformNoiseStart->value(), ui->spinBoxUniformNoiseStop->value());
    RandomGenUniformDistribution = new boost::variate_generator<boost::minstd_rand&, boost::uniform_int<>>(*rngUniformDist, *uniformDistribution);

    ModeSelect();
}

void MainWindow::on_checkBoxAddUniformNoise_toggled(bool checked)
{
    /*
    ready = 0;
    ui->checkBoxAddRician->setCheckState(Qt::Unchecked);
    ui->checkBoxAddNoise->setCheckState(Qt::Unchecked);
    ready = 1;
    */
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxRicianS_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxAddRician_toggled(bool checked)
{
    /*
    ready = 0;
    ui->checkBoxAddNoise->setCheckState(Qt::Unchecked);
    ui->checkBoxAddUniformNoise->setCheckState(Qt::Unchecked);
    ready = 1;
    */
    ModeSelect();
}

void MainWindow::on_checkBoxPlainImage_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_spinBoxRoiSize_valueChanged(int arg1)
{

    ui->spinBoxRoiShift->setMinimum(ui->spinBoxRoiSize->value());
    ui->spinBoxRoiOffset->setMinimum( ui->spinBoxRoiSize->value()/2);

    ModeSelect();
}

void MainWindow::on_spinBoxRoiOffset_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxRoiShift_valueChanged(const QString &arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxSkipCount_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxReducedROI_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_checkBoxSaveRoi_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxIntensityScale_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxGradDenominator_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxGradNominator_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxRoiNr_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxShowNormalisedROI_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxROIScale_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_pushButtonProcessAll_clicked()
{
    string CumulatedStatString = StatisticStringHeader();
    OutStringStat.clear();
    if (!exists(OutFolder))
    {
        ui->textEditOut->append( string("Error" + OutFolder.string() + " does not exists").c_str());
    }
    if (!is_directory(OutFolder))
    {
        ui->textEditOut->append(QString::fromStdString( string("Error 2" + OutFolder.string() + " is not a directory")));
    }

    ui->listWidgetImageFiles->currentRow()==1;
    OutString.clear();
    OutString = "";
    int filesCount = ui->listWidgetImageFiles->count();
    ui->textEditOut->clear();
    for(int fileNr = 0; fileNr< filesCount; fileNr++)
    {
        ui->listWidgetImageFiles->setCurrentRow(fileNr);
        CumulatedStatString += OutStringStat;
    }

    switch(operationMode)
    {
    case 3:
        {
            CumulatedStatString += OutStringStat;

            path textOutFile = OutFolder;
            textOutFile.append("HistStatistics.txt");

            std::ofstream out (textOutFile.string());
            out << CumulatedStatString;
            out.close();
        }
        break;
    case 4:
        {
            path textOutFile = OutFolder;
            textOutFile.append(ui->lineEditMaZdaScriptFileName->text().toStdString() + "_"+ ui->lineEditMaZdaOptionsFile->text().toStdString()+ ".bat");

            std::ofstream out (textOutFile.string());
            out << OutString;
            out.close();
        }
        break;

    default:

            break;
    }



}

void MainWindow::on_lineEditMaZdaOptionsFile_returnPressed()
{
    ui->listWidgetImageFiles->currentRow()==1;
    OutString.clear();
    OutString = "";
    int filesCount = ui->listWidgetImageFiles->count();
    ui->textEditOut->clear();
    for(int fileNr = 0; fileNr< filesCount; fileNr++)
    {
        ui->listWidgetImageFiles->setCurrentRow(fileNr);
    }
    path textOutFile = OutFolder;
    textOutFile.append(ui->lineEditMaZdaScriptFileName->text().toStdString() + "_"+ ui->lineEditMaZdaOptionsFile->text().toStdString()+ ".bat");

    std::ofstream out (textOutFile.string());
    out << OutString;
    out.close();
}

void MainWindow::on_checkBoxFixtRangeHistogram_toggled(bool checked)
{

    ModeSelect();
}

void MainWindow::on_spinBoxMinHist_valueChanged(int arg1)
{

    ModeSelect();
}

void MainWindow::on_spinBoxMaxHist_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxShowBinedROI_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_comboBoxROINorm_currentIndexChanged(int index)
{
    ModeSelect();
}

void MainWindow::on_spinBoxROIBitPerPix_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_HistogramBin_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_lineEditViewROIFolder_returnPressed()
{
    ModeSelect();
}

void MainWindow::on_checkBoxViewRoiShowBined_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_comboBoxViewROINorm_currentIndexChanged(int index)
{
    ModeSelect();
}

void MainWindow::on_spinBoxViewROIBitPerPixel_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxViewROINr_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxViewSaveBinnedROIImage_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_checkBoxVewSaveRoiBinnedHistogram_toggled(bool checked)
{
    ModeSelect();
}
