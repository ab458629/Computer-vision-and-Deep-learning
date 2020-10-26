/*
HW01_01.cpp
航太所碩一 P46091204 蔡承穎  Copyright (C) 2020
程式內容 : Camera Calibration
(1) Corner detection
(2) Find the intrinsic matrix
(3) Find the extrinsic matrix
(4) Find the distortion matrix
以下為使用opencv2函式的版本
*/

// HW01_01Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "HW01_01.h"
#include "HW01_01Dlg.h"
#include "afxdialogex.h"
#include <Windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// 對 App About 使用 CAboutDlg 對話方塊

using namespace std;
using namespace cv;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()

};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CHW01_01Dlg 對話方塊


CHW01_01Dlg::CHW01_01Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_HW01_01_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CHW01_01Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO1, m_ctrlComb);
}

BEGIN_MESSAGE_MAP(CHW01_01Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON6, &CHW01_01Dlg::OnBnClickedButton6)
	ON_CBN_SELCHANGE(IDC_COMBO1, &CHW01_01Dlg::OnCbnSelchangeCombo1)
	ON_BN_CLICKED(IDC_BUTTON4, &CHW01_01Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON7, &CHW01_01Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON5, &CHW01_01Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON8, &CHW01_01Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CHW01_01Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON11, &CHW01_01Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON10, &CHW01_01Dlg::OnBnClickedButton10)
END_MESSAGE_MAP()


// CHW01_01Dlg 訊息處理常式

BOOL CHW01_01Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	CString strName;
	strName.Empty();
	for (int i = 1; i <= 15; i++) {
		strName.Format(_T("%d"), i);
		m_ctrlComb.AddString(strName);
	}


	AllocConsole();
	FILE *stream;
	freopen_s(&stream, "CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CHW01_01Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void CHW01_01Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。

HCURSOR CHW01_01Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CHW01_01Dlg::OnCbnSelchangeCombo1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
}


void CHW01_01Dlg::OnBnClickedButton4()
{
	for (int i = 0; i< ChessBoard[1]; i++) {
		for (int j = 0; j < ChessBoard[0]; j++) {
			objp.push_back(Point3f(j, i, 0));
		}
	}
	vector<String> images;
	string path = "c:/Hw1/Q1_Image/*.bmp";
	glob(path, images);			// cv可以很方便的尋訪整個資料夾的檔案

	vector<Point2f> corners;

	bool success;
	namedWindow("Callibration", 0);
	resizeWindow("Callibration", 640, 480);
	for (int i = 0; i < images.size(); i++) {

		frame = imread(images[i]);

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		success = findChessboardCorners(gray, cvSize(ChessBoard[0], ChessBoard[1]), corners,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		/*
		vs. int found = cvFindChessboardCorners(image, board_sz, corners, &corner_count,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);	// 尋找內角點的子像素值 (儲存格式不同就要呼叫不同函式)
		*/

		if (success) {
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
			cornerSubPix(gray, corners, Size(11, 11), cv::Size(-1, -1), criteria);
			drawChessboardCorners(frame, Size(ChessBoard[0], ChessBoard[1]), corners, success);
			// vs. cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
			obj_points.push_back(objp);
			img_points.push_back(corners);
		}
		else if (success == 0) { continue; } // 若沒有找到直接下一個循環
		imshow("Callibration", frame);
		waitKey(500);
	}
	destroyAllWindows();
	calibrateCamera(obj_points, img_points, Size(gray.rows, gray.cols), intrinsics, distCoeffs, rvecs, tvecs);// calibrateCamera()給的世界座標方向是由obj_points的檢查順序與findChessboardCorners的檢查順序共同決定
}


void CHW01_01Dlg::OnBnClickedButton5()
{
	Ptr<Formatter> formatMat = Formatter::get(Formatter::FMT_DEFAULT);
	formatMat->set64fPrecision(6);

	stringstream stream;
	stream << formatMat->format(distCoeffs);
	cout << stream.str() << endl;
}


void CHW01_01Dlg::OnBnClickedButton6()
{
	int i = m_ctrlComb.GetCurSel();
	Rodrigues(rvecs[i], rmatrix);		// Lie Algebra SO(3) - > SE(3)
	Mat mat_tvec = Mat(tvecs[i]).clone();
	hconcat(rmatrix, mat_tvec, RT);

	Ptr<Formatter> formatMat = Formatter::get(Formatter::FMT_DEFAULT);
	formatMat->set64fPrecision(6);

	stringstream stream;
	stream << formatMat->format(RT);
	cout << stream.str() << endl;
}


void CHW01_01Dlg::OnBnClickedButton7()
{
	Ptr<Formatter> formatMat = Formatter::get(Formatter::FMT_DEFAULT);
	formatMat->set64fPrecision(8);

	stringstream stream;
	stream << formatMat->format(intrinsics);
	cout << stream.str() << endl;
}


void CHW01_01Dlg::OnBnClickedButton8()
{
	vector<vector<Point2f> > img_points;	// imagePoints: 其對應的圖像點，vector<vector<Point2f>>
	vector<vector<Point3f> > obj_points;	// objectPoints: 世界坐標系的，vector<vector<Point3f>>
	vector<Point3f> objp;

	for (int i = 0; i< ChessBoard[1]; i++) {
		for (int j = 0; j < ChessBoard[0]; j++) {
			objp.push_back(Point3f(j, i, 0));
		}
	}

	vector<String> images;
	string path = "c:/Hw1/Q2_Image/*.bmp";
	glob(path, images);			// cv可以很方便的尋訪整個資料夾的檔案
	Mat frame, gray;
	vector<Point2f> corners;
	bool success;

	for (int i = 0; i < images.size(); i++) {
		frame = imread(images[i]);

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		success = findChessboardCorners(gray, cvSize(ChessBoard[0], ChessBoard[1]), corners,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (success) {
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
			cornerSubPix(gray, corners, Size(11, 11), cv::Size(-1, -1), criteria);
			drawChessboardCorners(frame, Size(ChessBoard[0], ChessBoard[1]), corners, success);

			obj_points.push_back(objp);
			img_points.push_back(corners);
		}
		else if (success == 0) { continue; } // 若沒有找到直接下一個循環								 										 
	}


	Mat intrinsics, distCoeffs;
	vector< Mat > rvecs, tvecs;
	calibrateCamera(obj_points, img_points, Size(gray.rows, gray.cols), intrinsics, distCoeffs, rvecs, tvecs);	// calibrateCamera()給的世界座標方向是由obj_points的檢查順序與findChessboardCorners的檢查順序共同決定
																												// 由以上函式可以得到 intrinsics, distCoeffs

	Mat rmatrix, RT;
	namedWindow("Draw a tetrahedron", 0);
	resizeWindow("Draw a tetrahedron", 640, 480);

	vector<Point3f> Points;
	int x, y, z;

	x = 4; y = 3; z = -3;
	Points.push_back(Point3f(x, y, z));

	x = 6; y = 1; z = 0;
	Points.push_back(Point3f(x, y, z));

	x = 2; y = 3; z = 0;
	Points.push_back(Point3f(x, y, z));

	x = 6; y = 5; z = 0;
	Points.push_back(Point3f(x, y, z));

	Scalar colorLine(0, 0, 255);
	int thicknessLine = 5;

	vector<Point2f> undistortMatrix;

	for (int i = 0; i < images.size(); i++) {
		frame = imread(images[i]);
		Rodrigues(rvecs[i], rmatrix);
		// cout << "No. " << i + 1 << "Extrinsic Matrix : " << endl;
		Mat mat_tvec = Mat(tvecs[i]).clone();
		hconcat(rmatrix, mat_tvec, RT);	// extrinsic parameters [R|T]


		projectPoints(Points, rmatrix, tvecs[i], intrinsics, distCoeffs, undistortMatrix);

		line(frame, undistortMatrix[0], undistortMatrix[1], colorLine, thicknessLine);
		line(frame, undistortMatrix[0], undistortMatrix[2], colorLine, thicknessLine);
		line(frame, undistortMatrix[0], undistortMatrix[3], colorLine, thicknessLine);
		line(frame, undistortMatrix[1], undistortMatrix[2], colorLine, thicknessLine);
		line(frame, undistortMatrix[2], undistortMatrix[3], colorLine, thicknessLine);
		line(frame, undistortMatrix[3], undistortMatrix[1], colorLine, thicknessLine);

		imshow("Draw a tetrahedron", frame);
		waitKey(500);

		/* 以下可以計算誤差
		double err = 0.0;
		vector<Point2f>undistortMatrix;
		projectPoints(obj_points[i], rmatrix, tvecs[i], intrinsics, distCoeffs, undistortMatrix);
		err = norm(undistortMatrix, img_points[i]);
		cout << "第" << i << "張圖的誤差：" << err << endl;
		cout << undistortMatrix << endl;
		*/
	}
	destroyAllWindows();
}


void CHW01_01Dlg::OnBnClickedButton9()
{
	/*
	 (1) Block Matching（BM） StereoBM
     (2) Semi-Global Block Matching（SGBM） StereoSGBM
     (3) Graph Cut（GC）cvStereoGCState()
     (4) Dynamic Programming（DP）cvFindStereoCorrespondence()
	 BM演算法：速度很快，效果一般
	*/

	vector<String> img;
	string path = "c:/Hw1/Q3_Image/*.png";
	glob(path, img);			// cv可以很方便的尋訪整個資料夾的檔案

	Mat imgL = imread(img[0], IMREAD_GRAYSCALE);
	Mat imgR = imread(img[1], IMREAD_GRAYSCALE);
	Mat grayL, grayR;

	int mindisparity = 0;
	int ndisparities = 16;
	int SADWindowSize = 17;
	Ptr<StereoBM> bm = StereoBM::create(ndisparities, SADWindowSize);
	bm->setBlockSize(SADWindowSize);
	bm->setMinDisparity(mindisparity);
	bm->setNumDisparities(ndisparities);
	bm->setUniquenessRatio(15);
	// numDisparities：最大視差值與最小視差值之差，窗口大小必須是16的整數倍，int型
	// blockSize：匹配塊大小，必須是 >= 1的奇數，通常3~11

	copyMakeBorder(imgL, imgL, 0, 0, 80, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(imgR, imgR, 0, 0, 80, 0, IPL_BORDER_REPLICATE);
	Mat disparity;
	// Compute disparity
	bm->compute(imgL, imgR, disparity);

	// Convert
	disparity.convertTo(disparity, CV_32F, 1.0 / 16); //除以16得到真實視差值
	disparity = disparity.colRange(80, disparity.cols);
	Mat disp8U = Mat(disparity.rows, disparity.cols, CV_8UC1);

	normalize(disparity, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("Disparity map", disp8U);
}


void CHW01_01Dlg::OnBnClickedButton10()
{
	/*
	SIFT(尺度不變特徵變換，Scale-Invairiant Feature Transform)
	提取SIFT關鍵點(Key-point)，並計算SIFT描述子(Descriptor)
	屬於奢侈的圖像特徵方式，所以才有FAST或者ORB的出現
	*/ 


	SIFT_img1 = imread("c:/Hw1/Q4_Image/Aerial1.jpg", IMREAD_GRAYSCALE);
	SIFT_img2 = imread("c:/Hw1/Q4_Image/Aerial2.jpg", IMREAD_GRAYSCALE);

	Ptr<Feature2D> SIFT = xfeatures2d::SIFT::create();

	// 特徵擷取
	SIFT->detect(SIFT_img1, keypoints1);
	SIFT->detect(SIFT_img2, keypoints2);

	// 特徵描述子
	SIFT->compute(SIFT_img1, keypoints1, descriptors1);
	SIFT->compute(SIFT_img2, keypoints2, descriptors2);

	/*
	Mat outimg1;
	drawKeypoints(SIFT_img1, keypoints1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	Mat outimg2;
	drawKeypoints(SIFT_img2, keypoints2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("Keypoints1", outimg1);
	imshow("Keypoints2", outimg2);
	*/

	vector<DMatch> matches;
	BFMatcher matcher;
	matcher.match(descriptors1, descriptors2, matches);


	double min_dist = 25, max_dist = 0;	for (int i = 0; i < descriptors1.rows; i++){
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	vector< DMatch > good_matches;
	for (int i = 0; i < descriptors1.rows; i++){
		if (matches[i].distance <= max(2 * min_dist, 30.0)){
			good_matches.push_back(matches[i]);
		}
	}

	drawMatches(SIFT_img1, keypoints1, SIFT_img1, keypoints2, good_matches, img_match);

	/*
	RANSAC
	(1) 根據Matches將特徵點對齊，將座標轉換成float類型
	(2) 使用求基本矩陣方法findFundamentalMat，得到RansacStatus
	(3) 根據RansacStatus來將誤差匹配的點，即RansacStatus[i] = 0
	*/
	vector<DMatch> m_Matches;
	m_Matches = good_matches;

	// 座標轉換成float類型
	vector <KeyPoint> RAN_KP1, RAN_KP2;
	for (int i = 0; i < m_Matches.size(); i++) {
		RAN_KP1.push_back(keypoints1[good_matches[i].queryIdx]);
		RAN_KP2.push_back(keypoints2[good_matches[i].trainIdx]);
	}

	// 座標轉換
	vector <Point2f> p01, p02;
	for (int i = 0; i < m_Matches.size(); i++) {
		p01.push_back(RAN_KP1[i].pt);
		p02.push_back(RAN_KP2[i].pt);
	}

	// 求轉換矩陣
	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);

	vector <KeyPoint> RR_KP1, RR_KP2;
	vector <DMatch> RR_matches;
	int index = 0;

	for (int i = 0; i < m_Matches.size(); i++) {
		if (RansacStatus[i] != 0) {
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
			m_Matches[i].queryIdx = index;
			m_Matches[i].trainIdx = index;
			RR_matches.push_back(m_Matches[i]);
			index++;
		}
	}

	Mat outimg1;
	drawKeypoints(SIFT_img1, RR_KP1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	Mat outimg2;
	drawKeypoints(SIFT_img2, RR_KP2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("Keypoints1", outimg1);
	imshow("Keypoints2", outimg2);

	drawMatches(SIFT_img1, RR_KP1, SIFT_img2, RR_KP2, RR_matches, img_RR_matches);

}


void CHW01_01Dlg::OnBnClickedButton11()
{	imshow("Matches2", img_RR_matches);
}

