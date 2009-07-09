class BackgroundSubtractor
{
  public:
  BackgroundSubtractor(int width, int height, bool debug=false)
  {
    m_width = width;
    m_height = height;
    CvSize size = cvSize(m_width, m_height);
    m_debug = debug;
    m_source8u1_im = cvCreateImage(size, IPL_DEPTH_8U, 1);
    m_source32f1_im = cvCreateImage(size, IPL_DEPTH_32F, 1);
    m_acc32f1_im = cvCreateImage(size, IPL_DEPTH_32F, 1);
    m_bg32f1_im = cvCreateImage(size, IPL_DEPTH_32F, 1);
    m_mask32f1_im = cvCreateImage(size, IPL_DEPTH_32F, 1);

    if (m_debug) {
      cvNamedWindow("acc");
      cvNamedWindow("bg");
    }
  }

  void update(IplImage *source8u_im, IplImage *mask8u1_im, int thresh)
  {
    // updates mask to reflect changed pixels in source for a given thresh
    if (source8u_im->nChannels == 3) {
      cvCvtColor(source8u_im, m_source8u1_im, CV_BGR2GRAY);
    } else {
      m_source8u1_im = source8u_im;
    }
    cvCvtScale(m_source8u1_im, m_source32f1_im, 1/255.0);
    cvRunningAvg(m_source32f1_im, m_acc32f1_im, 0.1);

    cvAbsDiff(m_source32f1_im, m_bg32f1_im, m_mask32f1_im);
    cvCvtScale(m_mask32f1_im, mask8u1_im, 255);
    cvThreshold(mask8u1_im, mask8u1_im, thresh, 255, CV_THRESH_BINARY);

    if (m_debug) {
      cvShowImage("acc", m_acc32f1_im);
      cvShowImage("bg", m_bg32f1_im);
    }

  }

  void reset_bg()
  {
    cvCopy(m_acc32f1_im, m_bg32f1_im);
  }

  private:
  int m_width, m_height;
  bool m_debug;
  IplImage *m_source8u1_im, *m_source32f1_im, *m_acc32f1_im, *m_bg32f1_im, *m_mask32f1_im;
};

