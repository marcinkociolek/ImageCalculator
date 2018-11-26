#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    boost::filesystem::path ImageFolder;
    boost::filesystem::path OutFolder;

    std::string FileName;
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

    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();



    void OpenImageFolder();
    void ReadImage();
    void ShowsScaledImage(cv::Mat Im, std::string ImWindowName, double dispScale);
    void ModeSelect();
    void TiffRoiFromRed();
    void ImageResize();

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

private:
    Ui::MainWindow *ui;


};

#endif // MAINWINDOW_H
