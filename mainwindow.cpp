#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DispLib.h"
#include <tiffio.h>


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
    ready = 1;
}
//------------------------------------------------------------------------------------------------------------------------------
MainWindow::~MainWindow()
{
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
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName, double dispScale)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;

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
    if(ui->checkBoxShowOutput->checkState())
        ShowsScaledImage(ImOut, "Output Image", 1.0);

    if(ui->checkBoxShowOutMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImOut)));

}
//------------------------------------------------------------------------------------------------------------------------------

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
