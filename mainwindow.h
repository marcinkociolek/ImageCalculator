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
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    boost::filesystem::path ImageFolder;
    boost::filesystem::path OutFolder;

    void OpenImageFolder();


private slots:

    void on_pushButtonOpenImageFolder_clicked();

    void on_pushButtonOpenOutFolder_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
