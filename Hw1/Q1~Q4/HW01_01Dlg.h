
// HW01_01Dlg.h : ���Y��
//

#pragma once
#include "afxwin.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

// CHW01_01Dlg ��ܤ��
class CHW01_01Dlg : public CDialogEx
{
// �غc
public:
	CHW01_01Dlg(CWnd* pParent = NULL);	// �зǫغc�禡

// ��ܤ�����
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_HW01_01_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �䴩


// �{���X��@
protected:
	HICON m_hIcon;

	// ���ͪ��T�������禡
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButton6();
	afx_msg void OnCbnSelchangeCombo1();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton5();
	CComboBox m_ctrlComb;
	afx_msg void OnBnClickedButton8();

private:
	int ChessBoard[2]{ 8,11 };							// C++11�~���\�o��
	std::vector<std::vector<cv::Point3f> > obj_points;	// objectPoints: �@�ɧ��Шt���Avector<vector<Point3f>>
	std::vector<std::vector<cv::Point2f> > img_points;	// imagePoints: ��������Ϲ��I�Avector<vector<Point2f>>
	std::vector<cv::Point3f> objp;
	cv::Mat frame, gray;
	cv::Mat intrinsics, distCoeffs;
	std::vector< cv::Mat > rvecs, tvecs;
	cv::Mat rmatrix, RT;	cv::Mat SIFT_img1;
	cv::Mat SIFT_img2;	std::vector<cv::KeyPoint> keypoints1, keypoints2;	cv::Mat descriptors1, descriptors2;	cv::Mat img_match;
	cv::Mat img_RR_matches;

public:
	afx_msg void OnBnClickedButton9();
	afx_msg void OnBnClickedButton10();
	afx_msg void OnBnClickedButton11();
};
