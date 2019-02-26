#include "mv_init.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "CameraApi.h"
#include <semaphore.h>


using namespace std;


MvInit::MvInit(int cameraId)
{
	/* sem_init()第二个参数为0表示这个信号量是当前进程的局部信号量，否则该信号
	 * 就可以在多个进程之间共享 */
	if(sem_init(&sems, 0, 0) < 0)
	{
		cerr << "sems init err" << endl;
		exit(0);
	}
		
	createCamera(cameraId);
}


MvInit::~MvInit()
{
	
}
unsigned char  * g_pRgbBuffer;
static void* uiDisplayThread(void* lpParam)
{
	tSdkFrameHead 	sFrameInfo;

	MvInit          *pThis = (MvInit*)lpParam;
	BYTE*			pbyBuffer;
	CameraSdkStatus status = CAMERA_STATUS_SUCCESS;

	g_pRgbBuffer = (BYTE *)malloc(IMAGE_COLS*IMAGE_ROWS * 3);

	while (!pThis->m_bExit)
	{

		if (CameraGetImageBuffer(pThis->m_hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		{
			//将获得的原始数据转换成RGB格式的数据，同时经过ISP模块，对图像进行降噪，边沿提升，颜色校正等处理。
			//我公司大部分型号的相机，原始数据都是Bayer格式的
			status = CameraImageProcess(pThis->m_hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);//连续模式

																						 //分辨率改变了，则刷新背景
			if (IMAGE_COLS != sFrameInfo.iWidth || IMAGE_ROWS != sFrameInfo.iHeight)
			{
				cout << "cols:" << sFrameInfo.iWidth << "rows" << sFrameInfo.iHeight << endl;
				exit(0);
			}
			else
			{
				// 获得相机的参数信息
				memcpy(&pThis->m_sFrInfo, &sFrameInfo, sizeof(tSdkFrameHead));
			}

			if (status == CAMERA_STATUS_SUCCESS)
			{
				//调用SDK封装好的显示接口来显示图像,您也可以将m_pFrameBuffer中的RGB数据通过其他方式显示，比如directX,OpengGL,等方式。
				//CameraImageOverlay(hCamera, m_pFrameBuffer, &sFrameInfo);

				cv::Mat matImage(
					Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
					sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
					g_pRgbBuffer
				);
				Mat srcBGR;
				cvtColor(matImage,srcBGR,COLOR_BGR2RGB);
				srcBGR.copyTo(pThis->srcImage);
				sem_post(&pThis->sems);
			}

			//在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
			//否则再次调用CameraGetImageBuffer时，程序将被挂起，知道其他线程中调用CameraReleaseImageBuffer来释放了buffer
			CameraReleaseImageBuffer(pThis->m_hCamera, pbyBuffer);
		}
	}
	std::cout << "start release source...\r\n";
	CameraUnInit(pThis->m_hCamera);
	free(g_pRgbBuffer);
	sem_destroy(&pThis->sems);
	pthread_exit(0); 
	return NULL;
}


CameraSdkStatus MvInit::createCamera(int cameraId)
{
	
	tSdkCameraDevInfo sCameraList[CAMERA_NUM];
	INT iCameraNums;
	CameraSdkStatus status = CAMERA_STATUS_SUCCESS;
	tSdkCameraCapbility sCameraInfo;
	
	//枚举设备，获得设备列表
	iCameraNums = CAMERA_NUM;//调用CameraEnumerateDevice前，先设置iCameraNums = 10，表示最多只读取10个设备，如果需要枚举更多的设备，请更改sCameraList数组的大小和iCameraNums的值
	
	std::cout << "CameraEnumerateDevice" << std::endl;
	if (CameraEnumerateDevice(sCameraList, &iCameraNums) != CAMERA_STATUS_SUCCESS || iCameraNums == 0)
	{
		printf("No camera was found!");
		return FALSE;
	}
	 
	// 初始化相机
	std::cout << "CameraInit" << std::endl;
	if ((status = CameraInit(&sCameraList[cameraId], -1, -1, &m_hCamera)) != CAMERA_STATUS_SUCCESS)
	{
		char msg[128];
		//sprintf_s(msg, "Failed to init the camera! Error code is %d", status);
		snprintf(msg,sizeof(msg),"Failed to init the camera! Error code is %d", status);
		cout << msg << endl;
		//printf(msg);
		//printf(CameraGetErrorString(status));
		return FALSE;
	}

	//"获得该相机的特性描述"
	CameraGetCapability(m_hCamera, &sCameraInfo);

	std::cout << "set imageResolution" << std::endl;
    // ROI
	 tSdkImageResolution TSdkImageResolution;
	 /*通过tSdkImageResolution结构体中的iIndex编号进行
	 选定分辨率，iIndex = 0，1280*960
	 iIndex = 1 640*480 ROI center;
	 iIndex = 2 640*480 
	 others is about GTK_Demo	 
	 */
    TSdkImageResolution.iIndex = 0;
	// TSdkImageResolution.iWidth = 640;
	// TSdkImageResolution.iHeight = 480;
	CameraSetImageResolution(m_hCamera,&TSdkImageResolution);
    //tSdkImageResolution GetTSdkImageResolution;
	// CameraGetImageResolution(m_hCamera,&GetTSdkImageResolution);
	//  std::cout << GetTSdkImageResolution.iWidth << std::endl;
	//  std::cout << GetTSdkImageResolution.iHeight << std::endl;
	

	std::cout << "set exposureTime" << std::endl;
	// exposea// 设置曝光时间手动曝光
	CameraSetAeState(m_hCamera,FALSE);
    // 曝光时间us，尽量达到30帧每秒。
	CameraSetExposureTime(m_hCamera,30000);
	// 模拟增益
	CameraSetAnalogGain(m_hCamera,4);
	// RGB增益
	CameraSetGain(m_hCamera,112,100,137);
	// 饱和度
	CameraSetSaturation(m_hCamera,100);

	if (sCameraInfo.sIspCapacity.bMonoSensor)
	{
		CameraSetIspOutFormat(m_hCamera, CAMERA_MEDIA_TYPE_MONO8);
	}
	else
	{
        CameraSetIspOutFormat(m_hCamera,CAMERA_MEDIA_TYPE_RGB8);
    }

	strcpy(g_CameraName, sCameraList[0].acFriendlyName);

	// CameraCreateSettingPage(m_hCamera, NULL,
	// 	 g_CameraName, NULL, NULL, 0);//"通知SDK内部建该相机的属性页面";

     
	//m_hDispThread = (HANDLE)_beginthreadex(NULL, 0, &uiDisplayThread, (PVOID)this, 0, &m_threadID);
	
	CameraPlay(m_hCamera);

	//创建图像获取线程
	std::cout << "create thread for getting image" << std::endl;
	pthread_create(&id,NULL,uiDisplayThread,(void*)this);
	return TRUE;
	// CameraShowSettingPage(m_hCamera, FALSE);//TRUE显示相机配置界面。FALSE则隐藏。
}
