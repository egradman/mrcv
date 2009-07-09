#include <stdio.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#ifndef CAMERA_HPP
#define CAMERA_HPP

CvPoint2D32f g_click_point;
bool g_clicked=false;
void unprojectionMouseCallback(int event, int x, int y, int flags, void *param)
{
	if (event != CV_EVENT_LBUTTONDOWN) return;
  g_click_point.x = x;
  g_click_point.y = y;
  g_clicked = true;
}

class Camera
{
  public:
    Camera(bool debug=true)
    {
      m_debug = debug;

      if (m_debug) {
        cvNamedWindow("input");
      }
    }

    int loadIntrinsics(char *intrinsics_file, char *distortion_file)
    {
      captureInto(&m_temp);
      m_image = cvCloneImage(m_temp);

      m_intrinsics = (CvMat*)cvLoad(intrinsics_file);
      m_distortion = (CvMat*)cvLoad(distortion_file);

      if (!m_intrinsics || !m_distortion) {
        fprintf(stderr, "generating new intrinsics and distortion\n");
        calibrateIntrinsics(intrinsics_file, distortion_file);
        m_intrinsics = (CvMat*)cvLoad(intrinsics_file);
        m_distortion = (CvMat*)cvLoad(distortion_file);
      }

      // Build the undistort map which we will use for all 
      // subsequent frames.
      //
      m_mapx = cvCreateImage( cvGetSize(m_image), IPL_DEPTH_32F, 1 );
      m_mapy = cvCreateImage( cvGetSize(m_image), IPL_DEPTH_32F, 1 );
      cvInitUndistortMap(
        m_intrinsics,
        m_distortion,
        m_mapx,
        m_mapy
      );

      return 0;
    }

    int loadUnprojection(char *unprojection_file)
    {
      m_unprojection = (CvMat*)cvLoad(unprojection_file);
      if (!m_unprojection) {
        calibrateUnprojection(unprojection_file);
        m_unprojection = (CvMat*)cvLoad(unprojection_file);
      }
      return 0;
    }

    void finalizeImage(void)
    {
      if (m_debug) {
        cvShowImage("input", m_image);
      }
    }

    virtual void captureInto(IplImage **dest) = 0;

    IplImage *getRawFrame(void)
    {
      captureInto(&m_image);
      finalizeImage();
      return m_image;
    }

    IplImage *getRectifiedFrame(void)
    {
      captureInto(&m_temp);
      cvRemap( m_temp, m_image, m_mapx, m_mapy );
      finalizeImage();
      return m_image;
    }

    IplImage *getUnprojectedFrame(void)
    {
      static IplImage *unproject_temp;
      captureInto(&m_temp);
      cvRemap( m_temp, m_image, m_mapx, m_mapy );
      if (!unproject_temp) {
        unproject_temp = cvCloneImage(m_image);
      }

      cvCopyImage(m_image, unproject_temp);
      cvWarpPerspective(unproject_temp, m_image, m_unprojection, CV_WARP_FILL_OUTLIERS, cvScalar(255,255,255));
      finalizeImage();
      return m_image;
    }

  protected:
    IplImage *m_image;
    IplImage *m_temp;

    bool m_debug;

    IplImage *m_mapx, *m_mapy;
    CvMat *m_intrinsics, *m_distortion;
    CvMat *m_unprojection;

  private:
    int calibrateIntrinsics(char *intrinsics_file, char *distortion_file)
    {
      int board_w  = 6;
      int board_h  = 8;
      int n_boards = 25;
      int board_dt = 30;
      
      int board_n  = board_w * board_h;
      CvSize board_sz = cvSize( board_w, board_h );

      cvNamedWindow( "Calibration" );
      cvNamedWindow( "Raw Video");
      //
      //ALLOCATE STORAGE
      CvMat* image_points      = cvCreateMat(n_boards*board_n,2,CV_32FC1);
      CvMat* object_points     = cvCreateMat(n_boards*board_n,3,CV_32FC1);
      CvMat* point_counts      = cvCreateMat(n_boards,1,CV_32SC1);
      CvMat* intrinsic_matrix  = cvCreateMat(3,3,CV_32FC1);
      CvMat* distortion_coeffs = cvCreateMat(4,1,CV_32FC1);

      CvPoint2D32f* corners = new CvPoint2D32f[ board_n ];
      int corner_count;
      int successes = 0;
      int step, frame = 0;

      IplImage *image;
      captureInto(&image);
      IplImage *gray_image = cvCreateImage(cvGetSize(image),8,1);
     
      // CAPTURE CORNER VIEWS LOOP UNTIL WE GOT n_boards 
      // SUCCESSFUL CAPTURES (ALL CORNERS ON THE BOARD ARE FOUND)
      //
      while(successes < n_boards) {
        //Skip every board_dt frames to allow user to move chessboard
        if((frame++ % board_dt) == 0) {
           //Find chessboard corners:
           int found = cvFindChessboardCorners(
                    image, board_sz, corners, &corner_count, 
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
           );

           //Get Subpixel accuracy on those corners
           cvCvtColor(image, gray_image, CV_BGR2GRAY);
           cvFindCornerSubPix(gray_image, corners, corner_count, 
                      cvSize(11,11),cvSize(-1,-1), cvTermCriteria(    
                      CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

           //Draw it
           cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);
       
           // If we got a good board, add it to our data
           if( corner_count == board_n ) {
              cvShowImage( "Calibration", image ); //show in color if we did collect the image
              step = successes*board_n;
              for( int i=step, j=0; j<board_n; ++i,++j ) {
                 CV_MAT_ELEM(*image_points, float,i,0) = corners[j].x;
                 CV_MAT_ELEM(*image_points, float,i,1) = corners[j].y;
                 CV_MAT_ELEM(*object_points,float,i,0) = j/board_w;
                 CV_MAT_ELEM(*object_points,float,i,1) = j%board_w;
                 CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
              }
              CV_MAT_ELEM(*point_counts, int,successes,0) = board_n;    
              successes++;
              printf("Collected our %d of %d needed chessboard images\n",successes,n_boards);
           }
           else
             cvShowImage( "Calibration", gray_image ); //Show Gray if we didn't collect the image
        } //end skip board_dt between chessboard capture

        //Handle pause/unpause and ESC
        int c = cvWaitKey(15);
        if(c == 'p'){  
           c = 0;
           while(c != 'p' && c != 27){
                c = cvWaitKey(250);
           }
         }
         if(c == 27)
            return 0;
        captureInto(&image);
        cvShowImage("Raw Video", image);
      } //END COLLECTION WHILE LOOP.
      cvDestroyWindow("Calibration");
      printf("\n\n*** CALLIBRATING THE CAMERA...");
      //ALLOCATE MATRICES ACCORDING TO HOW MANY CHESSBOARDS FOUND
      CvMat* object_points2  = cvCreateMat(successes*board_n,3,CV_32FC1);
      CvMat* image_points2   = cvCreateMat(successes*board_n,2,CV_32FC1);
      CvMat* point_counts2   = cvCreateMat(successes,1,CV_32SC1);
      //TRANSFER THE POINTS INTO THE CORRECT SIZE MATRICES
      for(int i = 0; i<successes*board_n; ++i){
          CV_MAT_ELEM( *image_points2, float, i, 0) = 
                 CV_MAT_ELEM( *image_points, float, i, 0);
          CV_MAT_ELEM( *image_points2, float,i,1) =   
                 CV_MAT_ELEM( *image_points, float, i, 1);
          CV_MAT_ELEM(*object_points2, float, i, 0) =  
                 CV_MAT_ELEM( *object_points, float, i, 0) ;
          CV_MAT_ELEM( *object_points2, float, i, 1) = 
                 CV_MAT_ELEM( *object_points, float, i, 1) ;
          CV_MAT_ELEM( *object_points2, float, i, 2) = 
                 CV_MAT_ELEM( *object_points, float, i, 2) ;
      } 
      for(int i=0; i<successes; ++i){ //These are all the same number
        CV_MAT_ELEM( *point_counts2, int, i, 0) = 
                 CV_MAT_ELEM( *point_counts, int, i, 0);
      }
      cvReleaseMat(&object_points);
      cvReleaseMat(&image_points);
      cvReleaseMat(&point_counts);

      // At this point we have all of the chessboard corners we need.
      // Initialize the intrinsic matrix such that the two focal
      // lengths have a ratio of 1.0
      //
      CV_MAT_ELEM( *intrinsic_matrix, float, 0, 0 ) = 1.0f;
      CV_MAT_ELEM( *intrinsic_matrix, float, 1, 1 ) = 1.0f;

      //CALIBRATE THE CAMERA!
      cvCalibrateCamera2(
          object_points2, image_points2,
          point_counts2,  cvGetSize( image ),
          intrinsic_matrix, distortion_coeffs,
          NULL, NULL,0  //CV_CALIB_FIX_ASPECT_RATIO
      );

      // SAVE THE INTRINSICS AND DISTORTIONS
      printf(" *** DONE!\n\nStoring intrinsics and Distortion files\n\n");
      cvSave(intrinsics_file,intrinsic_matrix);
      cvSave(distortion_file,distortion_coeffs);

      return 0;
    }

  void calibrateUnprojection(char *unprojection_file)
  {
    IplImage *image;
    CvPoint2D32f src[4];
    CvPoint2D32f dst[4];
    int n_points = 0;

    fprintf(stderr, "calibrating new unprojection\n");

    cvNamedWindow("calibrateUnprojection");
    cvSetMouseCallback("calibrateUnprojection", unprojectionMouseCallback);

    while (n_points < 4) {
      printf("click point %d\n", n_points);
      //switch (n_points) {
      //  case 0:
      //    cvRectangle(g_im_output, cvPoint(0,0), cvPoint(m_image->width/2, m_image->height/2), cvScalar(0,0,255), -1);
      //    break;
      //  case 1:
      //    cvRectangle(g_im_output, cvPoint(m_image->width,0), cvPoint(m_image->width/2, m_image->height/2), cvScalar(0,0,255), -1);
      //    break;
      //  case 2:
      //    cvRectangle(g_im_output, cvPoint(m_image->width,m_image->height), cvPoint(m_image->width/2, m_image->height/2), cvScalar(0,0,255), -1);
      //    break;
      //  case 3:
      //    cvRectangle(g_im_output, cvPoint(0,m_image->height), cvPoint(m_image->width/2, m_image->height/2), cvScalar(0,0,255), -1);
      //    break;
      //}

      while (!g_clicked) {
        image = getRectifiedFrame();
        cvShowImage("calibrateUnprojection", image);
        cvWaitKey(20);
      }
      g_clicked = false;
      src[n_points] = g_click_point;
      //for (int i=1; i<=n_points; i++) {
      //  cvLine(g_im_input, cvPoint(src[i-1].x, src[i-1].y), cvPoint(src[i].x, src[i].y), cvScalar(255,0,0));
      //}
      //cvShowImage("input", g_im_input);

      n_points++;
    }

    dst[0] = cvPoint2D32f(0, 0);
    dst[1] = cvPoint2D32f(m_image->width, 0);
    dst[2] = cvPoint2D32f(m_image->width, m_image->height);
    dst[3] = cvPoint2D32f(0, m_image->height);
    
    m_unprojection = cvCreateMat(3, 3, CV_32FC1);
    cvGetPerspectiveTransform(src, dst, m_unprojection);
    cvSave(unprojection_file,m_unprojection);
  }
};

class RegularCamera: public Camera
{
  public:
    RegularCamera(int id, int width=640, int height=480, int fps=30, bool debug=true) : Camera(debug)
    {
      m_id = id;
      m_capture = cvCreateCameraCapture(id);
      cvSetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_WIDTH, width);
      cvSetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_HEIGHT, height);
      cvSetCaptureProperty(m_capture, CV_CAP_PROP_FPS, fps);
      if (!m_capture) {
        fprintf(stderr, "could not open regular camera %d!\n", m_id);
        return;
      }
    }

  protected:
    void captureInto(IplImage **dest)
    {
      *dest = cvQueryFrame(m_capture);
    }
  private:
    int m_id;
  public:
    CvCapture *m_capture;

};

#endif
