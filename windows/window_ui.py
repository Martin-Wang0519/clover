# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/rc/four-leaves.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        MainWindow.setIconSize(QtCore.QSize(32, 32))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setStyleSheet("")
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_6.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.addPosPicButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.addPosPicButton.setFont(font)
        self.addPosPicButton.setAutoFillBackground(True)
        self.addPosPicButton.setStyleSheet("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/rc/add_pic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addPosPicButton.setIcon(icon1)
        self.addPosPicButton.setIconSize(QtCore.QSize(48, 48))
        self.addPosPicButton.setObjectName("addPosPicButton")
        self.gridLayout.addWidget(self.addPosPicButton, 0, 0, 1, 1)
        self.addNegPicButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.addNegPicButton.setFont(font)
        self.addNegPicButton.setAutoFillBackground(True)
        self.addNegPicButton.setStyleSheet("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/rc/add_neg-pic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addNegPicButton.setIcon(icon2)
        self.addNegPicButton.setIconSize(QtCore.QSize(48, 48))
        self.addNegPicButton.setObjectName("addNegPicButton")
        self.gridLayout.addWidget(self.addNegPicButton, 1, 0, 1, 1)
        self.posLcdNumber = QtWidgets.QLCDNumber(self.centralWidget)
        self.posLcdNumber.setObjectName("posLcdNumber")
        self.gridLayout.addWidget(self.posLcdNumber, 0, 2, 1, 1)
        self.negLcdNumber = QtWidgets.QLCDNumber(self.centralWidget)
        self.negLcdNumber.setObjectName("negLcdNumber")
        self.gridLayout.addWidget(self.negLcdNumber, 1, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("font: 14pt \"楷体\";")
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("font: 14pt \"楷体\";")
        self.label_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.trainButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.trainButton.setFont(font)
        self.trainButton.setAutoFillBackground(True)
        self.trainButton.setStyleSheet("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/rc/train.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.trainButton.setIcon(icon3)
        self.trainButton.setIconSize(QtCore.QSize(48, 48))
        self.trainButton.setObjectName("trainButton")
        self.verticalLayout.addWidget(self.trainButton)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_4.setContentsMargins(11, 11, 9, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setStyleSheet("font: 14pt \"楷体\";")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.aButton = QtWidgets.QRadioButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aButton.sizePolicy().hasHeightForWidth())
        self.aButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.aButton.setFont(font)
        self.aButton.setStyleSheet("font: 14pt \"楷体\";")
        self.aButton.setObjectName("aButton")
        self.horizontalLayout_4.addWidget(self.aButton)
        self.gangButton = QtWidgets.QRadioButton(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gangButton.sizePolicy().hasHeightForWidth())
        self.gangButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.gangButton.setFont(font)
        self.gangButton.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.gangButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.gangButton.setStyleSheet("font: 14pt \"楷体\";")
        self.gangButton.setObjectName("gangButton")
        self.horizontalLayout_4.addWidget(self.gangButton)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_10 = QtWidgets.QLabel(self.groupBox_4)
        self.label_10.setStyleSheet("font: 14pt \"楷体\";")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_3.addWidget(self.label_10)
        self.shotFrequencySpinBox = QtWidgets.QSpinBox(self.groupBox_4)
        self.shotFrequencySpinBox.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.shotFrequencySpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.shotFrequencySpinBox.setMinimum(1)
        self.shotFrequencySpinBox.setMaximum(5)
        self.shotFrequencySpinBox.setObjectName("shotFrequencySpinBox")
        self.horizontalLayout_3.addWidget(self.shotFrequencySpinBox)
        self.horizontalLayout_2.addWidget(self.groupBox_4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mailEdit = QtWidgets.QTextEdit(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mailEdit.sizePolicy().hasHeightForWidth())
        self.mailEdit.setSizePolicy(sizePolicy)
        self.mailEdit.setStyleSheet("font: 14pt \"Times New Roman\";")
        self.mailEdit.setLineWidth(1)
        self.mailEdit.setObjectName("mailEdit")
        self.horizontalLayout.addWidget(self.mailEdit)
        self.horizontalLayout_5.addLayout(self.horizontalLayout)
        self.label_6 = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("font: 14pt \"楷体\";")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.screenPredictButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.screenPredictButton.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/rc/screenshot.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.screenPredictButton.setIcon(icon4)
        self.screenPredictButton.setIconSize(QtCore.QSize(48, 48))
        self.screenPredictButton.setObjectName("screenPredictButton")
        self.verticalLayout_2.addWidget(self.screenPredictButton)
        self.testButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.testButton.setFont(font)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/rc/test.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.testButton.setIcon(icon5)
        self.testButton.setIconSize(QtCore.QSize(48, 48))
        self.testButton.setObjectName("testButton")
        self.verticalLayout_2.addWidget(self.testButton)
        self.winningCalculationButton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(28)
        self.winningCalculationButton.setFont(font)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/rc/win.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.winningCalculationButton.setIcon(icon6)
        self.winningCalculationButton.setIconSize(QtCore.QSize(48, 48))
        self.winningCalculationButton.setObjectName("winningCalculationButton")
        self.verticalLayout_2.addWidget(self.winningCalculationButton)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.textBrowser.setFont(font)
        self.textBrowser.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.textBrowser.setStyleSheet("font: 12pt \"Times New Roman\";")
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_2.addWidget(self.textBrowser)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menuBar.setStyleSheet("")
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.action_line = QtWidgets.QAction(MainWindow)
        self.action_line.setEnabled(True)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/rc/Line.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_line.setIcon(icon7)
        self.action_line.setObjectName("action_line")
        self.action_triangle = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/rc/Triangle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_triangle.setIcon(icon8)
        self.action_triangle.setVisible(False)
        self.action_triangle.setIconVisibleInMenu(True)
        self.action_triangle.setObjectName("action_triangle")
        self.action_rectangle = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/rc/Rectangle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_rectangle.setIcon(icon9)
        self.action_rectangle.setObjectName("action_rectangle")
        self.action_circle = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/rc/Circle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_circle.setIcon(icon10)
        self.action_circle.setObjectName("action_circle")
        self.action_ellipse = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/rc/Ellipse.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_ellipse.setIcon(icon11)
        self.action_ellipse.setVisible(False)
        self.action_ellipse.setObjectName("action_ellipse")
        self.action_polygon = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/rc/Octagon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_polygon.setIcon(icon12)
        self.action_polygon.setObjectName("action_polygon")
        self.action_palette = QtWidgets.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/rc/Palette.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_palette.setIcon(icon13)
        self.action_palette.setObjectName("action_palette")
        self.action_translate = QtWidgets.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/rc/Translate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_translate.setIcon(icon14)
        self.action_translate.setObjectName("action_translate")
        self.action_trash = QtWidgets.QAction(MainWindow)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(":/rc/Trash.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_trash.setIcon(icon15)
        self.action_trash.setObjectName("action_trash")
        self.action_rotate = QtWidgets.QAction(MainWindow)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(":/rc/Rotate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_rotate.setIcon(icon16)
        self.action_rotate.setVisible(False)
        self.action_rotate.setObjectName("action_rotate")
        self.action_zoomin = QtWidgets.QAction(MainWindow)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap(":/rc/ZoomIn.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_zoomin.setIcon(icon17)
        self.action_zoomin.setVisible(False)
        self.action_zoomin.setObjectName("action_zoomin")
        self.action_zoomout = QtWidgets.QAction(MainWindow)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap(":/rc/ZoomOut.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_zoomout.setIcon(icon18)
        self.action_zoomout.setVisible(False)
        self.action_zoomout.setObjectName("action_zoomout")
        self.action_save = QtWidgets.QAction(MainWindow)
        self.action_save.setObjectName("action_save")
        self.action_open = QtWidgets.QAction(MainWindow)
        icon19 = QtGui.QIcon()
        icon19.addPixmap(QtGui.QPixmap(":/rc/Open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_open.setIcon(icon19)
        self.action_open.setObjectName("action_open")
        self.action_clip = QtWidgets.QAction(MainWindow)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap(":/rc/Clip.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_clip.setIcon(icon20)
        self.action_clip.setVisible(False)
        self.action_clip.setObjectName("action_clip")
        self.action_curve = QtWidgets.QAction(MainWindow)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(QtGui.QPixmap(":/rc/Curve.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_curve.setIcon(icon21)
        self.action_curve.setVisible(False)
        self.action_curve.setObjectName("action_curve")
        self.action_addpoint = QtWidgets.QAction(MainWindow)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(QtGui.QPixmap(":/rc/AddPoint.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_addpoint.setIcon(icon22)
        self.action_addpoint.setVisible(False)
        self.action_addpoint.setObjectName("action_addpoint")
        self.action_deletepoint = QtWidgets.QAction(MainWindow)
        icon23 = QtGui.QIcon()
        icon23.addPixmap(QtGui.QPixmap(":/rc/DeletePoint.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_deletepoint.setIcon(icon23)
        self.action_deletepoint.setVisible(False)
        self.action_deletepoint.setObjectName("action_deletepoint")
        self.action_pre = QtWidgets.QAction(MainWindow)
        icon24 = QtGui.QIcon()
        icon24.addPixmap(QtGui.QPixmap(":/rc/pre.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_pre.setIcon(icon24)
        self.action_pre.setObjectName("action_pre")
        self.action_next = QtWidgets.QAction(MainWindow)
        icon25 = QtGui.QIcon()
        icon25.addPixmap(QtGui.QPixmap(":/rc/next.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_next.setIcon(icon25)
        self.action_next.setObjectName("action_next")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "股票预测"))
        self.addPosPicButton.setToolTip(_translate("MainWindow", "导入模型数据"))
        self.addPosPicButton.setText(_translate("MainWindow", "导入背离图片"))
        self.addNegPicButton.setToolTip(_translate("MainWindow", "导入模型数据"))
        self.addNegPicButton.setText(_translate("MainWindow", "导入非背离图片"))
        self.label_11.setText(_translate("MainWindow", "当前背离图片总数"))
        self.label_12.setText(_translate("MainWindow", "当前非背离图片总数"))
        self.trainButton.setToolTip(_translate("MainWindow", "导入模型数据"))
        self.trainButton.setText(_translate("MainWindow", "开始训练"))
        self.label_4.setText(_translate("MainWindow", "股票类型"))
        self.aButton.setText(_translate("MainWindow", "a股"))
        self.gangButton.setText(_translate("MainWindow", "港股"))
        self.label_10.setText(_translate("MainWindow", "设置每秒截图数"))
        self.label_6.setText(_translate("MainWindow", "输入邮箱，多个邮箱请分行输入"))
        self.screenPredictButton.setToolTip(_translate("MainWindow", "截图并预测"))
        self.screenPredictButton.setText(_translate("MainWindow", "截图并预测"))
        self.testButton.setToolTip(_translate("MainWindow", "截图并预测"))
        self.testButton.setText(_translate("MainWindow", "测试"))
        self.winningCalculationButton.setToolTip(_translate("MainWindow", "截图并预测"))
        self.winningCalculationButton.setText(_translate("MainWindow", "胜率统计"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Times New Roman\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'SimSun\'; font-size:9pt;\"><br /></p></body></html>"))
        self.action_line.setText(_translate("MainWindow", "直线"))
        self.action_triangle.setText(_translate("MainWindow", "三角形"))
        self.action_rectangle.setText(_translate("MainWindow", "矩形"))
        self.action_circle.setText(_translate("MainWindow", "圆"))
        self.action_ellipse.setText(_translate("MainWindow", "椭圆"))
        self.action_polygon.setText(_translate("MainWindow", "多边形"))
        self.action_palette.setText(_translate("MainWindow", "调色板"))
        self.action_translate.setText(_translate("MainWindow", "移动"))
        self.action_trash.setText(_translate("MainWindow", "删除"))
        self.action_rotate.setText(_translate("MainWindow", "旋转"))
        self.action_zoomin.setText(_translate("MainWindow", "放大"))
        self.action_zoomout.setText(_translate("MainWindow", "缩小"))
        self.action_save.setText(_translate("MainWindow", "保存"))
        self.action_open.setText(_translate("MainWindow", "打开"))
        self.action_clip.setText(_translate("MainWindow", "裁剪"))
        self.action_curve.setText(_translate("MainWindow", "曲线"))
        self.action_addpoint.setText(_translate("MainWindow", "加粗"))
        self.action_deletepoint.setText(_translate("MainWindow", "减细"))
        self.action_pre.setText(_translate("MainWindow", "上一张"))
        self.action_pre.setToolTip(_translate("MainWindow", "上一张"))
        self.action_next.setText(_translate("MainWindow", "下一张"))
        self.action_next.setToolTip(_translate("MainWindow", "下一张"))
import windows.rc_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
