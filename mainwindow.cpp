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

    ShowsScaledImage(ImIn, "Input Image PC", displayScale, ui->comboBoxDisplayRange->currentIndex());
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

    double minDisp = 0.0;
    double maxDisp = 255.0;

    switch(dispMode)
    {
    case 1:
        minDisp = ui->doubleSpinBoxFixMinDisp->value();
        maxDisp = ui->doubleSpinBoxFixMaxDisp->value();

        break;
    case 2:
        NormParamsMinMax(Im, &maxDisp, &minDisp);
        break;
    case 3:
        NormParamsMeanP3Std(Im, &maxDisp, &minDisp);
        break;
    case 4:
        NormParams1to99perc(Im, &maxDisp, &minDisp);
        break;
    default:
        break;
    }

    if(dispMode > 0)
    {
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
        ImIn32S = Mat::ones(ImIn.size(), CV_32S) * (int32_t)round(ui->doubleSpinBoxIntOffset->value());
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
                double valIm = *wImOut;
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
                int val;
                switch(ui->comboBoxGradientDirection->currentIndex())
                {
                case 1:
                    val = y * ui->spinBoxGradientNominator->value()/ui->spinBoxGradientDenominator->value();
                    break;
                case 2:
                    val = (x + y) * ui->spinBoxGradientNominator->value()/ui->spinBoxGradientDenominator->value();
                    break;
                default:
                    val = x * ui->spinBoxGradientNominator->value()/ui->spinBoxGradientDenominator->value();
                    break;
                }

                *wImOut += (int32_t)val;
                wImOut ++;
            }
        }
    }

    ImOut = ImOut + (int32_t)round(ui->doubleSpinBoxIntOffset->value());


    if(ui->checkBoxShowOutput->checkState())
        ShowsScaledImage(ImOut, "Output Image", displayScale,ui->comboBoxDisplayRange->currentIndex());

    ImOut.convertTo(ImOut,CV_16U,1.0,0.0);

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
            OutFileName += to_string(ui->doubleSpinBoxGaussNianoiseSigma->value());
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



    int roiNr = 1;
    int skip = 0;

    for (int y = firstRoiY; y < lastRoiY; y += roiShift)
    {
        for (int x = firstRoiX; x < lastRoiX; x += roiShift)
        {
            if(ui->checkBoxReducedROI->checkState())
            {
                if (skip <= 0)
                    skip = ui->spinBoxSkipCount->value();
                else
                {
                    skip--;
                    continue;
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
    ModeSelect();
}

void MainWindow::on_checkBoxAddGradient_toggled(bool checked)
{
    ModeSelect();
}

void MainWindow::on_spinBoxGradientNominator_valueChanged(int arg1)
{
    ModeSelect();
}

void MainWindow::on_spinBoxGradientDenominator_valueChanged(int arg1)
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
    ModeSelect();
}

void MainWindow::on_doubleSpinBoxRicianS_valueChanged(double arg1)
{
    ModeSelect();
}

void MainWindow::on_checkBoxAddRician_toggled(bool checked)
{
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
