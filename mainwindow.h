#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <boost/filesystem.hpp>


#include <opencv2/core/core.hpp>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    static boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<>>* RandomGen1();

public:
    boost::filesystem::path ImageFolder;
    boost::filesystem::path OutFolder;

    std::string FileName;

    std::string OutString;

    cv::Mat ImIn;
    cv::Mat ImOut;

    int operationMode;

    double displayScale;
    double xPixelSize;

    bool showInImage;

    double resizeScale;

    double xPixSizeOut;

    int resizeInterpolation;

    bool ready;

    double minIm;
    double maxIm;

    boost::minstd_rand* rngNormalDist;
    boost::normal_distribution<>* normalDistribution;
    boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<>>* RandomGenNormDistribution;

    boost::minstd_rand* rngUniformDist;
    boost::uniform_int<>* uniformDistribution;
    boost::variate_generator<boost::minstd_rand&, boost::uniform_int<>>* RandomGenUniformDistribution;

    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();



    void OpenImageFolder();
    void ReadImage();
    void ShowsScaledImage(cv::Mat Im, std::string ImWindowName, double dispScale, int dispMode = 0);
    void ShowsScaledImage(cv::Mat Im, cv::Mat Mask, std::string ImWindowName, double dispScale, uint16_t RoiNr, int dispMode );
    void SaveScaledImage(cv::Mat Im, std::string FileName, double dispScale,int dispMode);
    void SaveScaledImage(cv::Mat Im, cv::Mat Mask, std::string FileName, double dispScale, uint16_t RoiNr, int dispMode );
    void ModeSelect();
    void TiffRoiFromRed();
    void ImageResize();
    void ImageLinearOperation();
    void CreateROI();
    std::string CreateMaZdaScript();
    //void GetDisplayParams(Mat ImIn, double maxIm, double minIm);

private slots:

    void on_pushButtonOpenImageFolder_clicked();

    void on_pushButtonOpenOutFolder_clicked();

    void on_lineEditRegexImageFile_returnPressed();

    void on_listWidgetImageFiles_currentTextChanged(const QString &currentText);

    void on_spinBoxScaleBase_valueChanged(int arg1);

    void on_spinBoxScalePower_valueChanged(int arg1);

    void on_tabWidgetMode_currentChanged(int index);




    void on_lineEditImageScale_returnPressed();

    void on_comboBoxImageInterpolationMethod_currentIndexChanged(int index);

    void on_checkBoxShowOutput_toggled(bool checked);

    void on_pushButtonSaveResized_clicked();

    void on_lineEditPixelSize_returnPressed();

    void on_checkBoxLoadAnydepth_toggled(bool checked);

    void on_checkBoxSaveOutput_toggled(bool checked);

    void on_comboBoxDisplayRange_currentIndexChanged(int index);

    void on_doubleSpinBoxIntOffset_valueChanged(double arg1);

    void on_doubleSpinBoxFixMinDisp_valueChanged(double arg1);

    void on_doubleSpinBoxFixMaxDisp_valueChanged(double arg1);

    void on_checkBoxShowHist_stateChanged(int arg1);

    void on_spinBoxHistBarWidth_valueChanged(int arg1);

    void on_spinBoxHistScaleHeight_valueChanged(int arg1);

    void on_spinBoxHistScaleCoef_valueChanged(int arg1);

    void on_checkBoxAddNoise_stateChanged(int arg1);

    void on_doubleSpinBoxGaussNianoiseSigma_valueChanged(double arg1);

    void on_checkBoxAddNoise_toggled(bool checked);

    void on_checkBoxAddGradient_toggled(bool checked);

    void on_comboBoxGradientDirection_currentIndexChanged(int index);

    void on_spinBoxUniformNoiseStart_valueChanged(int arg1);

    void on_spinBoxUniformNoiseStop_valueChanged(int arg1);

    void on_checkBoxAddUniformNoise_toggled(bool checked);

    void on_doubleSpinBoxRicianS_valueChanged(double arg1);

    void on_checkBoxAddRician_toggled(bool checked);

    void on_checkBoxPlainImage_toggled(bool checked);

    void on_spinBoxRoiSize_valueChanged(int arg1);

    void on_spinBoxRoiOffset_valueChanged(int arg1);

    void on_spinBoxRoiShift_valueChanged(const QString &arg1);

    void on_spinBoxSkipCount_valueChanged(int arg1);

    void on_checkBoxReducedROI_toggled(bool checked);

    void on_checkBoxSaveRoi_toggled(bool checked);

    void on_doubleSpinBoxIntensityScale_valueChanged(double arg1);

    void on_doubleSpinBoxGradDenominator_valueChanged(double arg1);

    void on_doubleSpinBoxGradNominator_valueChanged(double arg1);

    void on_spinBoxRoiNr_valueChanged(int arg1);

    void on_checkBoxShowNormalisedROI_toggled(bool checked);

    void on_doubleSpinBoxROIScale_valueChanged(double arg1);

    void on_pushButtonProcessAll_clicked();

private:
    Ui::MainWindow *ui;


};

#endif // MAINWINDOW_H
